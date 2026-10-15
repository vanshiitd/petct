#!/usr/bin/env python3
"""Build 4-channel prediction inputs for the refinement network (strategy 7).

The network was trained with channels 2 and 3 taken from *single-model* maps:
out-of-fold predictions for the 738 training cases, and `uncertainty_m0L` for
the 76 validation cases. At apply time there are two defensible choices, and
Phase IIIb is the reason to run both rather than pick one:

  ensemble   the 5-model ensemble maps, which is what the task specifies and
             what a deployed system would have
  single     the single-model maps, which match the distribution the network
             was trained on -- Phase IIIb measured U_stab on ensemble maps at
             roughly half its single-model value

Reporting both separates "the refinement network does not work" from "the
refinement network was fed a distribution it never saw".

PET and CT are hard-linked from Dataset505 rather than copied.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SRC = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")


def src_images(case: str) -> tuple[Path, Path]:
    for sub in ("imagesTs", "imagesTr"):
        p = SRC / sub / f"{case}_0000.nii.gz"
        if p.exists():
            return p, SRC / sub / f"{case}_0001.nii.gz"
    raise FileNotFoundError(case)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.maps.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} cases from {args.maps} -> {args.out}", flush=True)
    nlink = 0
    for i, f in enumerate(files, 1):
        case = f.stem
        pet, ct = src_images(case)
        for s, d in ((pet, args.out / f"{case}_0000.nii.gz"),
                     (ct, args.out / f"{case}_0001.nii.gz")):
            if not d.exists():
                try:
                    os.link(s, d)
                    nlink += 1
                except OSError:
                    shutil.copy2(s, d)
        ref = sitk.ReadImage(str(pet))
        z = np.load(f)
        shape = tuple(ref.GetSize()[::-1])
        if z["p_bar"].shape != shape:
            raise SystemExit(f"{case}: map {z['p_bar'].shape} != image {shape}")
        for ch, key in ((2, "p_bar"), (3, "u_stab")):
            img = sitk.GetImageFromArray(z[key].astype(np.float32))
            img.CopyInformation(ref)
            sitk.WriteImage(img, str(args.out / f"{case}_{ch:04d}.nii.gz"), True)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)
    print(f"done ({nlink} hard links)")


if __name__ == "__main__":
    main()
