#!/usr/bin/env python3
"""Preprocess the AutoPET dataset into (2, Z, Y, X) .npy volumes for pretraining.

Replaces: preprocessing/autopetproc.ipynb

Pipeline per scan: resample to a common spacing -> crop to the body ->
z-score each modality -> stack PET and CT as two channels.

Example:
    python scripts/preprocess_autopet.py \\
        --source /data/AutoPET2025_FDG --target /data/PETCTfoundation/AutoPET
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config  # noqa: E402
from petct.volume_ops import (  # noqa: E402
    get_body_bbox, normalize_volume, resample_to_shape, resample_to_spacing,
)


def process_scan(pet_path: Path, ct_path: Path, target_spacing_zyx: np.ndarray):
    """Return the stacked (2, Z, Y, X) array for one scan."""
    pet_img = sitk.ReadImage(str(pet_path))
    ct_img = sitk.ReadImage(str(ct_path))

    spacing_xyz = np.round(pet_img.GetSpacing()).astype(float)
    orig_spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])

    pet = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)

    pet_r = resample_to_spacing(pet, orig_spacing_zyx, target_spacing_zyx, order=1)
    ct_r = resample_to_shape(ct, pet_r.shape, order=1)

    bbox = get_body_bbox(
        ct_r, threshold=config.CT_BODY_THRESHOLD, pad=config.CROP_PAD, shape=pet_r.shape
    )
    if bbox is not None:
        pet_r, ct_r = pet_r[bbox], ct_r[bbox]

    return np.stack([normalize_volume(pet_r), normalize_volume(ct_r)], axis=0), pet_img.GetSpacing()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="root containing scan folders")
    p.add_argument("--target", type=Path, required=True, help="output root for .npy volumes")
    p.add_argument("--spacing", type=float, nargs=3, default=list(config.TARGET_SPACING_ZYX),
                   metavar=("Z", "Y", "X"), help="target voxel spacing in mm")
    p.add_argument("--overwrite", action="store_true", help="reprocess scans already present")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")

    target_spacing = np.array(args.spacing, dtype=float)
    print(f"AutoPET preprocessing\n  source: {args.source}\n  target: {args.target}\n"
          f"  spacing (z,y,x): {tuple(target_spacing)}\n")

    n_ok = n_skip = n_err = 0
    for pet_path in sorted(args.source.rglob("PET.nii.gz")):
        scan_dir = pet_path.parent
        ct_path = scan_dir / "CT_resample.nii.gz"
        if not ct_path.exists():
            ct_path = scan_dir / "CT.nii.gz"
        if not ct_path.exists():
            print(f"  [skip] no CT beside {scan_dir.name}")
            n_skip += 1
            continue

        rel = scan_dir.relative_to(args.source)
        save_path = args.target / rel / "PET.npy"
        if save_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            print(f"Processing {rel}")
            stacked, orig_spacing = process_scan(pet_path, ct_path, target_spacing)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, stacked)
            print(f"  spacing(xyz) {orig_spacing} -> shape {stacked.shape} -> {save_path}")
            n_ok += 1
        except Exception as e:  # keep going; one bad scan shouldn't stop the batch
            print(f"  [error] {scan_dir.name}: {e}")
            n_err += 1

    print(f"\nDone. processed={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
