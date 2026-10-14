#!/usr/bin/env python3
"""Build Dataset506: the refinement network's dataset (plan strategy 7, eqs. 21-22).

Four input channels, in this order:

    _0000  PET, prepared exactly as the MAE corpus was      (hard link from 505)
    _0001  CT, likewise                                     (hard link from 505)
    _0002  M0 probability  P-bar
    _0003  flip-stability uncertainty  U_stab

Target is the same ground truth as Dataset505, hard-linked.

PET, CT and the label are *hard links* into Dataset505's raw folder, not copies:
the bytes are identical by construction and the two extra channels then cost
about 1.8 MB per case instead of duplicating 40 GB. Both datasets sit on the
same volume, which is what makes that possible.

Which maps supply channels 2 and 3 matters more than it looks. For the 738
training cases the only non-leaking source is the out-of-fold prediction -- the
single fold model that did not train on that patient. For the 76 validation
cases *any* fold model is legitimate, since none of them trained on val, but the
saved val maps are the 5-model ensemble, and Phase IIIb measured U_stab on
ensemble maps at about half its single-model value. Selecting on a distribution
the network never sees in training would be a quiet mistake, so validation
channels come from the single-model `uncertainty_m0L` maps instead, which match
the out-of-fold distribution. The choice is recorded in the dataset description.

Geometry is copied from the Dataset505 PET image of the same case, so the four
channels agree voxel for voxel; the script refuses a case whose map shape does
not match that image rather than writing a misaligned channel.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SRC = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_channel(arr: np.ndarray, ref: sitk.Image, dst: Path) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(dst), True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(r"C:\nnunet_raw\Dataset506_AutoPET_Refine"))
    ap.add_argument("--train-maps", type=Path, default=Path(r"D:\data\cv\oof"))
    ap.add_argument("--val-maps", type=Path,
                    default=Path(r"D:\data\uncertainty_m0L\val"))
    ap.add_argument("--splits", type=Path,
                    default=Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    imtr = args.out / "imagesTr"
    latr = args.out / "labelsTr"
    for d in (imtr, latr):
        d.mkdir(parents=True, exist_ok=True)

    sets = [("train", args.train_maps), ("val", args.val_maps)]
    cases, stats = [], {"link": 0, "copy": 0, "exists": 0}
    for which, maps in sets:
        files = sorted(maps.glob("*.npz"))
        if args.limit:
            files = files[: args.limit]
        print(f"{which}: {len(files)} cases from {maps}", flush=True)
        for i, f in enumerate(files, 1):
            case = f.stem
            pet = SRC / "imagesTr" / f"{case}_0000.nii.gz"
            ct = SRC / "imagesTr" / f"{case}_0001.nii.gz"
            lab = SRC / "labelsTr" / f"{case}.nii.gz"
            if not pet.exists():
                raise SystemExit(f"{case}: no PET at {pet}")
            for s, d in ((pet, imtr / f"{case}_0000.nii.gz"),
                         (ct, imtr / f"{case}_0001.nii.gz"),
                         (lab, latr / f"{case}.nii.gz")):
                stats[link_or_copy(s, d)] += 1
            ref = sitk.ReadImage(str(pet))
            z = np.load(f)
            shape = tuple(ref.GetSize()[::-1])
            if z["p_bar"].shape != shape:
                raise SystemExit(f"{case}: map {z['p_bar'].shape} != image {shape}")
            write_channel(z["p_bar"], ref, imtr / f"{case}_0002.nii.gz")
            write_channel(z["u_stab"], ref, imtr / f"{case}_0003.nii.gz")
            cases.append((case, which))
            if i % 50 == 0 or i == len(files):
                print(f"  {which} {i}/{len(files)}", flush=True)

    desc = ("Refinement dataset for Phase IV strategy 7. Channels: 0 PET, 1 CT, "
            "2 M0 probability, 3 U_stab. PET/CT/labels hard-linked from "
            "Dataset505_AutoPET_MAEprep_v2. Channels 2-3 for the 738 training "
            "cases come from out-of-fold predictions (single held-out fold "
            "model); for the 76 validation cases from the single-model "
            "uncertainty_m0L maps, chosen over the saved 5-model ensemble maps "
            "so that selection sees the same U_stab distribution as training.")
    (args.out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "noNorm", "1": "noNorm", "2": "noNorm",
                          "3": "noNorm"},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "dataset_name": args.out.name,
        "description": desc}, indent=1))

    # fold 0 must be exactly train-on-738, select-on-76
    tr = sorted(c for c, w in cases if w == "train")
    va = sorted(c for c, w in cases if w == "val")
    (args.out / "splits_final_intended.json").write_text(
        json.dumps([{"train": sorted(tr), "val": sorted(va)}], indent=1))

    print(f"\n{len(cases)} cases -> {args.out}")
    print(f"  hard links {stats['link']}, copies {stats['copy']}, "
          f"already present {stats['exists']}")
    print(f"  fold 0: {len(tr)} train / {len(va)} val "
          f"-> splits_final_intended.json (copy into preprocessed after planning)")


if __name__ == "__main__":
    main()
