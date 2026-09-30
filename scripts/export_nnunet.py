#!/usr/bin/env python3
"""Export the frozen split as an nnU-Net v2 raw dataset.

nnU-Net wants one flat directory of channel-suffixed images plus a dataset.json:

    imagesTr/<case>_0000.nii.gz   CT     -- channel name "CT" so nnU-Net applies
    imagesTr/<case>_0001.nii.gz   PET       its CT normalisation (clip to the
    labelsTr/<case>.nii.gz        mask      foreground intensity percentiles and
    imagesTs/<case>_0000.nii.gz            z-score with dataset-wide statistics)
    labelsTs/<case>.nii.gz        kept aside for our own evaluation

Train *and* val cases both go in imagesTr -- nnU-Net splits them itself, and we
hand it our split via splits_final.json so fold 0 reproduces exactly the
partition in splits/autopet_v1.json. The test cases go to imagesTs and are never
seen during training; their labels live in labelsTs, which nnU-Net ignores.

Files are hardlinked when the source and target are on one volume, so the export
costs no extra disk and the originals are never moved or modified.

    python scripts/export_nnunet.py --data-root D:\\data\\autopet_nifti \\
        --split splits/autopet_v1.json --nnunet-raw D:\\data\\nnunet\\raw
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DATASET_ID = 501
DATASET_NAME = "AutoPET"


def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink where possible (same volume), else copy."""
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def scan_dir_for_case(data_root: Path, patient: str, case_id: str) -> Path:
    """Map a case id back to its scan directory.

    <PatientID> for a single-study patient, <PatientID>_<study_folder> for a
    multi-study one -- the patient prefix is stripped to recover the study part.
    """
    if case_id == patient:
        return data_root / patient
    study = case_id[len(patient) + 1:]
    return data_root / patient / study


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--nnunet-raw", type=Path, required=True)
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--dataset-name", default=DATASET_NAME)
    p.add_argument("--pet-file", default="PET.nii.gz",
                   help="which PET representation to export as channel 1 "
                        "(PET.nii.gz = raw Bq/mL, SUV.nii.gz = SUVbw)")
    p.add_argument("--pet-channel-name", default="PET",
                   help="channel_names entry for channel 1, which is what selects nnU-Net's "
                        "normalisation: 'PET' (or any unknown name) gives the default "
                        "per-image z-score; 'CT' gives CTNormalization, a dataset-wide "
                        "percentile clip and z-score, i.e. the same transform for every "
                        "patient; 'noNorm' leaves the data alone")
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    ds_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if ds_dir.exists() and any(ds_dir.iterdir()):
        raise SystemExit(f"{ds_dir} already exists and is not empty -- "
                         f"remove it or pass a different --dataset-id")

    modes = {"train": "Tr", "val": "Tr", "test": "Ts"}
    counts: dict[str, int] = {"link": 0, "copy": 0, "exists": 0}
    per_set: dict[str, int] = {}
    case_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for set_name, suffix in modes.items():
        cases = split["splits"][set_name]["cases"]
        n = 0
        for patient, case_ids in sorted(cases.items()):
            for case_id in case_ids:
                src = scan_dir_for_case(args.data_root, patient, case_id)
                for src_name, chan in (("CT_resample.nii.gz", "_0000"),
                                       (args.pet_file, "_0001")):
                    counts[link_or_copy(src / src_name,
                                        ds_dir / f"images{suffix}" / f"{case_id}{chan}.nii.gz")] += 1
                counts[link_or_copy(src / "tumorSeg.nii.gz",
                                    ds_dir / f"labels{suffix}" / f"{case_id}.nii.gz")] += 1
                case_lists[set_name].append(case_id)
                n += 1
        per_set[set_name] = n
        print(f"{set_name:5s}: {n} cases -> images{suffix}/labels{suffix}")

    n_training = per_set["train"] + per_set["val"]
    dataset_json = {
        # The channel NAME is what selects the normalisation scheme, so it is a
        # pipeline setting rather than a label. "CT" means CTNormalization:
        # clip to the dataset-wide foreground 0.5/99.5 percentiles, then z-score
        # with dataset-wide mean and std -- the identical transform for every
        # patient. The default for any other name is a per-image z-score, which
        # rescales each scan by its own statistics and so destroys the
        # cross-patient calibration that makes PET uptake comparable.
        "channel_names": {"0": "CT", "1": args.pet_channel_name},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "dataset_name": f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        "description": (
            "AutoPET whole-body FDG PET/CT lesion segmentation. Study-paired "
            "conversion of TCIA FDG-PET-CT-Lesions; split frozen in "
            f"{args.split.name} (seed {split['seed']}, patient-level). "
            f"Channel 1 is {args.pet_file}, normalised as "
            f"'{args.pet_channel_name}'."),
        "reference": "https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=1))

    # nnU-Net reads this from nnUNet_preprocessed, not raw, but writing it here
    # keeps the export self-contained; the caller copies it across after
    # preprocessing has created that directory.
    splits_final = [{"train": case_lists["train"], "val": case_lists["val"]}]
    (ds_dir / "splits_final.json").write_text(json.dumps(splits_final, indent=1))

    print(f"\nfiles: {counts['link']} hardlinked, {counts['copy']} copied, "
          f"{counts['exists']} already present")
    print(f"numTraining = {n_training}  (train {per_set['train']} + val {per_set['val']})")
    print(f"imagesTs    = {per_set['test']} cases (labels in labelsTs, for our own evaluation)")
    print(f"\nwritten to {ds_dir}")
    print(f"splits_final.json holds one fold: "
          f"{len(case_lists['train'])} train / {len(case_lists['val'])} val")
    print("Copy it into nnUNet_preprocessed/<dataset>/ after plan_and_preprocess, "
          "before training.")


if __name__ == "__main__":
    main()
