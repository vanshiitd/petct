#!/usr/bin/env python3
"""Preprocess the DeepPSMA dataset into (2, Z, Y, X) .npy volumes for pretraining.

Replaces: preprocessing/DeepPSMA copy.ipynb

Unlike the AutoPET/SPADE scripts (which resample with scipy.ndimage.zoom), this
one resamples through SimpleITK onto an explicit reference grid. That respects
the images' origin and direction cosines, so PET and CT stay physically aligned
rather than merely ending up the same array shape -- worth keeping, since
DeepPSMA's PET and CT are acquired on genuinely different grids.

CT is resampled with a -1024 HU fill value (air), not 0, so padding introduced
outside the original field of view isn't mistaken for soft tissue by the
body-cropping threshold.

Expected layout:  <source>/<batch>/<case>/FDG/{PET,CT}.nii.gz

Example:
    python scripts/preprocess_deeppsma.py \\
        --source /data/DeepPSMA --target /data/PETCTfoundation/DeepPSMA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config  # noqa: E402
from petct.volume_ops import get_body_bbox, normalize_volume  # noqa: E402


def make_reference_grid(pet_img: sitk.Image, target_spacing_xyz):
    """Build an empty image defining the target grid: PET's frame, new spacing."""
    orig_size = np.array(pet_img.GetSize(), dtype=np.int32)
    orig_spacing = np.array(pet_img.GetSpacing(), dtype=np.float32)
    target_spacing = np.array(target_spacing_xyz, dtype=np.float32)

    new_size = np.maximum(np.round(orig_size * orig_spacing / target_spacing).astype(np.int32), 1)

    ref = sitk.Image([int(v) for v in new_size], pet_img.GetPixelID())
    ref.SetSpacing(tuple(float(v) for v in target_spacing))
    ref.SetOrigin(pet_img.GetOrigin())
    ref.SetDirection(pet_img.GetDirection())
    return ref, orig_spacing, target_spacing, new_size


def resample_to_reference(img, ref, interpolator, default_value=0.0):
    return sitk.Resample(img, ref, sitk.Transform(), interpolator, default_value, img.GetPixelID())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--spacing", type=float, nargs=3, default=[2.0, 2.0, 3.0],
                   metavar=("X", "Y", "Z"), help="target spacing in mm, SimpleITK (x,y,z) order")
    p.add_argument("--pet-norm", choices=["z-score", "min-max"], default="z-score")
    p.add_argument("--ct-norm", choices=["z-score", "min-max"], default="z-score")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")

    pet_paths = sorted(args.source.rglob("FDG/PET.nii.gz"))
    print(f"DeepPSMA preprocessing\n  source: {args.source}\n  target: {args.target}\n"
          f"  found {len(pet_paths)} PET volumes\n")

    n_ok = n_skip = n_err = 0
    for pet_path in pet_paths:
        ct_path = pet_path.parent / "CT.nii.gz"
        if not ct_path.exists():
            print(f"  [skip] missing CT beside {pet_path.parent}")
            n_skip += 1
            continue

        rel = pet_path.relative_to(args.source)
        save_path = (args.target / rel).with_suffix("").with_suffix(".npy")
        if save_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            print(f"Processing {rel}")
            pet_img = sitk.ReadImage(str(pet_path))
            ct_img = sitk.ReadImage(str(ct_path))

            ref, orig_spacing, target_spacing, new_size = make_reference_grid(pet_img, args.spacing)
            pet_r = sitk.GetArrayFromImage(
                resample_to_reference(pet_img, ref, sitk.sitkLinear, 0.0)
            ).astype(np.float32)
            ct_r = sitk.GetArrayFromImage(
                resample_to_reference(ct_img, ref, sitk.sitkLinear, -1024.0)
            ).astype(np.float32)

            bbox = get_body_bbox(
                ct_r, threshold=config.CT_BODY_THRESHOLD, pad=config.CROP_PAD, shape=pet_r.shape
            )
            if bbox is not None:
                pet_r, ct_r = pet_r[bbox], ct_r[bbox]

            stacked = np.stack(
                [normalize_volume(pet_r, args.pet_norm), normalize_volume(ct_r, args.ct_norm)],
                axis=0,
            ).astype(np.float32)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, stacked)
            print(f"  spacing(xyz) {tuple(orig_spacing.tolist())} -> {tuple(target_spacing.tolist())}"
                  f"  shape {stacked.shape} -> {save_path}")
            n_ok += 1
        except Exception as e:
            print(f"  [error] {pet_path}: {e}")
            n_err += 1

    print(f"\nDone. processed={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
