#!/usr/bin/env python3
"""Export the frozen split prepared EXACTLY as the MAE pretraining corpus was.

The pretrained weights were learned on volumes that went through
`AutoPETPreprocessd`: resample to 3x2x2 mm, crop to the body, z-score each
modality over the cropped volume, PET first and CT second. Feeding nnU-Net's own
preprocessing instead would show those weights a different distribution in a
different channel order, and the pretraining would be worth little.

So this runs that same transform -- the actual class, not a copy of it -- and
writes the result as an nnU-Net raw dataset:

    imagesTr/<case>_0000.nii.gz   prepared PET   (channel 0, as in pretraining)
    imagesTr/<case>_0001.nii.gz   prepared CT    (channel 1)
    labelsTr/<case>.nii.gz        label, cropped the same way
    imagesTs/, labelsTs_prep/     the 200 test cases

`dataset.json` declares both channels `noNorm`: the data is already z-scored and
nnU-Net must not normalise it again.

Every case also gets `inverse/<case>.json`, holding the crop box and shapes
needed to put a prediction back on the original scan's grid --
`map_back_predictions.py` consumes it.

Geometry: with this dataset's spacing the resampling step is the identity (the
source spacing rounds to exactly the target), so the prepared volume is a plain
crop of the original voxel grid. The written images therefore keep the *true*
source spacing and direction, with the origin moved to the crop corner, and the
mapping back is exact rather than approximate.

    python scripts/export_nnunet_maeprep.py --data-root D:\\data\\autopet_nifti \\
        --split splits/autopet_v1.json --nnunet-raw C:\\nnunet_raw
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from petct.transforms import AutoPETPreprocessd  # noqa: E402

DATASET_ID = 503
DATASET_NAME = "AutoPET_MAEprep"


def scan_dir_for_case(data_root: Path, patient: str, case_id: str) -> Path:
    if case_id == patient:
        return data_root / patient
    return data_root / patient / case_id[len(patient) + 1:]


def write_like(array: np.ndarray, reference: sitk.Image, bbox, out_path: Path,
               dtype) -> None:
    """Save `array` on the original grid, shifted to the crop corner."""
    img = sitk.GetImageFromArray(array.astype(dtype))
    img.SetSpacing(reference.GetSpacing())
    img.SetDirection(reference.GetDirection())
    if bbox is None:
        img.SetOrigin(reference.GetOrigin())
    else:
        # bbox is (z, y, x); TransformIndexToPhysicalPoint takes (x, y, z)
        (z0, _), (y0, _), (x0, _) = bbox
        img.SetOrigin(reference.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def export_case(job) -> dict:
    case_id, scan_dir, ds_dir, suffix, label_dir, pet_file, pet_mu, pet_sigma = job
    scan_dir, ds_dir = Path(scan_dir), Path(ds_dir)

    tf = AutoPETPreprocessd(keys=["image", "label"], record_meta=True)
    out = tf({
        "pet_path": str(scan_dir / "PET.nii.gz"),
        "ct_path": str(scan_dir / "CT_resample.nii.gz"),
        "seg_path": str(scan_dir / "tumorSeg.nii.gz"),
    })
    meta = out["prep_meta"]
    image, label = out["image"], out["label"][0]
    bbox = meta["crop_bbox_zyx"]

    if not np.allclose(meta["zoom_zyx"], 1.0):
        # The inverse mapping below assumes the voxel grid was untouched. Rather
        # than write a volume whose geometry is quietly wrong, refuse it.
        return {"case": case_id, "status": "resampled",
                "detail": f"zoom {meta['zoom_zyx']} is not 1; inverse mapping would be approximate"}

    ref = sitk.ReadImage(str(scan_dir / "PET.nii.gz"))

    pet_channel = image[0]          # per-scan z-scored PET, as in pretraining
    if pet_mu is not None:
        # One transform for every case instead of per-scan z-scoring. The volume
        # is cropped with the SAME box the transform just computed, so the two
        # channels stay voxel-aligned; only the normalisation differs.
        alt = sitk.GetArrayFromImage(sitk.ReadImage(str(scan_dir / pet_file))).astype(np.float32)
        if bbox is not None:
            (z0, z1), (y0, y1), (x0, x1) = bbox
            alt = alt[z0:z1, y0:y1, x0:x1]
        if alt.shape != pet_channel.shape:
            return {"case": case_id, "status": "shape_mismatch",
                    "detail": f"{pet_file} crops to {alt.shape}, expected {pet_channel.shape}"}
        pet_channel = (alt - np.float32(pet_mu)) / np.float32(pet_sigma)

    write_like(pet_channel, ref, bbox, ds_dir / f"images{suffix}" / f"{case_id}_0000.nii.gz", np.float32)
    write_like(image[1], ref, bbox, ds_dir / f"images{suffix}" / f"{case_id}_0001.nii.gz", np.float32)
    write_like(label, ref, bbox, ds_dir / label_dir / f"{case_id}.nii.gz", np.uint8)

    meta["source_pet"] = str(scan_dir / "PET.nii.gz")
    meta["source_seg"] = str(scan_dir / "tumorSeg.nii.gz")
    meta["case"] = case_id
    (ds_dir / "inverse").mkdir(parents=True, exist_ok=True)
    (ds_dir / "inverse" / f"{case_id}.json").write_text(json.dumps(meta, indent=1))

    return {"case": case_id, "status": "ok",
            "shape": meta["prepared_shape_zyx"], "lesion": int(label.sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--nnunet-raw", type=Path, required=True)
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--dataset-name", default=DATASET_NAME)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="export only the first N cases (sizing run)")
    p.add_argument("--pet-file", default=None,
                   help="use this volume for channel 0 instead of the per-scan z-scored PET, "
                        "e.g. SUV.nii.gz. Requires --pet-mu and --pet-sigma.")
    p.add_argument("--pet-mu", type=float, default=None,
                   help="global mean for the PET channel: (x - mu) / sigma, the SAME for every "
                        "case, replacing the per-scan z-score. From scripts/suv_global_stats.py.")
    p.add_argument("--pet-sigma", type=float, default=None)
    args = p.parse_args()

    if (args.pet_file is None) != (args.pet_mu is None):
        raise SystemExit("--pet-file and --pet-mu/--pet-sigma must be given together")
    if args.pet_mu is not None and not args.pet_sigma:
        raise SystemExit("--pet-sigma must be given and non-zero")

    split = json.loads(args.split.read_text())
    ds_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    jobs, case_lists = [], {"train": [], "val": [], "test": []}
    for set_name, (suffix, label_dir) in (("train", ("Tr", "labelsTr")),
                                          ("val", ("Tr", "labelsTr")),
                                          ("test", ("Ts", "labelsTs_prep"))):
        for patient, case_ids in sorted(split["splits"][set_name]["cases"].items()):
            for case_id in case_ids:
                jobs.append((case_id, str(scan_dir_for_case(args.data_root, patient, case_id)),
                             str(ds_dir), suffix, label_dir,
                             args.pet_file, args.pet_mu, args.pet_sigma))
                case_lists[set_name].append(case_id)

    if args.limit:
        jobs = jobs[: args.limit]
        print(f"--limit {args.limit}: exporting {len(jobs)} cases only (sizing run)\n")
    print(f"exporting {len(jobs)} cases to {ds_dir}\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(export_case, jobs, chunksize=2), 1):
            results.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")

    bad = [r for r in results if r["status"] != "ok"]
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nexported {len(ok)} | refused {len(bad)}")
    for r in bad[:10]:
        print(f"   {r['case']}: {r['detail']}")
    if bad:
        raise SystemExit("refusing to continue with a partial export")

    if args.limit:
        total = sum(f.stat().st_size for f in ds_dir.rglob("*.nii.gz"))
        print(f"\n{len(ok)} cases occupy {total/1e9:.2f} GB "
              f"-> {total/len(ok)/1e6:.1f} MB per case")
        print(f"projected for {len(case_lists['train'])+len(case_lists['val'])+len(case_lists['test'])} "
              f"cases: {total/len(ok)*1014/1e9:.0f} GB")
        return

    n_training = len(case_lists["train"]) + len(case_lists["val"])
    (ds_dir / "dataset.json").write_text(json.dumps({
        # The volumes are already z-scored per channel by AutoPETPreprocessd,
        # which is what the pretrained weights expect. noNorm keeps nnU-Net from
        # normalising them a second time.
        "channel_names": {"0": "noNorm", "1": "noNorm"},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "dataset_name": f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        "description": (
            "AutoPET prepared exactly as the MAE pretraining corpus: 3x2x2 mm, "
            "body-cropped, per-channel z-score, channel 0 PET and channel 1 CT. "
            f"Split {args.split.name} (seed {split['seed']})."),
    }, indent=1))

    (ds_dir / "splits_final.json").write_text(json.dumps(
        [{"train": case_lists["train"], "val": case_lists["val"]}], indent=1))

    if args.pet_mu is not None:
        (ds_dir / "pet_normalisation.json").write_text(json.dumps({
            "channel_0_source": args.pet_file,
            "scheme": "global affine, identical for every case",
            "formula": "(x - mu) / sigma",
            "mu": args.pet_mu,
            "sigma": args.pet_sigma,
            "fitted_on": "training cases only, voxels inside the body crop",
            "channel_1_source": "CT_resample.nii.gz, per-scan z-score (unchanged)",
        }, indent=1))

    shapes = np.array([r["shape"] for r in ok])
    print(f"prepared shape (z,y,x): min {shapes.min(0)} | median {np.median(shapes,0).astype(int)} "
          f"| max {shapes.max(0)}")
    total = sum(f.stat().st_size for f in ds_dir.rglob("*.nii.gz"))
    print(f"on disk: {total/1e9:.1f} GB")
    print(f"numTraining = {n_training} (train {len(case_lists['train'])} + val {len(case_lists['val'])})")
    print(f"imagesTs = {len(case_lists['test'])}")
    print(f"\nwritten to {ds_dir}")


if __name__ == "__main__":
    main()
