#!/usr/bin/env python3
"""Write the ensemble segmentation (P_bar > 0.5) as masks in the prepared grid.

The ensemble is the 2-model x 8-flip average from `make_uncertainty_maps.py`.
Thresholding P_bar at 0.5 gives the segmentation the plan asks to be scored
alongside the single models, so it has to come out in the same grid and with the
same geometry as everything else -- which means borrowing the geometry from the
prepared image rather than inventing it, so `map_back_predictions.py` and the
evaluator accept these files unchanged.

    python scripts/export_ensemble_masks.py --maps D:\\data\\uncertainty\\val \\
        --out D:\\data\\nnunet\\predictions\\val_505ens_prep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")


def reference_image(case: str) -> sitk.Image:
    """Geometry comes from the prepared PET channel, never from thin air."""
    for sub in ("imagesTs", "imagesTr"):
        f = RAW / sub / f"{case}_0000.nii.gz"
        if f.exists():
            return sitk.ReadImage(str(f))
    raise FileNotFoundError(f"no prepared image for {case}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.maps.glob("*.npz"))
    print(f"{len(files)} maps -> {args.out}")

    n_pos = 0
    for i, f in enumerate(files, 1):
        case = f.stem
        p_bar = np.load(f)["p_bar"].astype(np.float32)
        mask = (p_bar > args.threshold).astype(np.uint8)
        ref = reference_image(case)
        arr_shape = sitk.GetArrayFromImage(ref).shape
        if mask.shape != arr_shape:
            raise SystemExit(f"{case}: map {mask.shape} != prepared image {arr_shape}")
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(ref)      # same grid by construction, verified above
        sitk.WriteImage(img, str(args.out / f"{case}.nii.gz"), True)
        n_pos += int(mask.any())
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    print(f"written {len(files)} | non-empty {n_pos}")


if __name__ == "__main__":
    main()
