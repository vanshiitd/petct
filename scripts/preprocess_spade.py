#!/usr/bin/env python3
"""Preprocess the SPADE dataset into (2, Z, Y, X) .npy volumes for pretraining.

SPADE arrives as pre-extracted pet.npy / ct.npy pairs in PatientID/ScanID/
folders, with a known fixed acquisition spacing, so there is no DICOM/NIfTI
metadata to read -- the source spacing is supplied as a flag.

Example:
    python scripts/preprocess_spade.py \\
        --source /data/spade --target /data/PETCTfoundation/Spade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config  # noqa: E402
from petct.volume_ops import (  # noqa: E402
    get_body_bbox, normalize_volume, resample_to_shape, resample_to_spacing,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--source-spacing", type=float, nargs=3, default=[3.0, 3.5, 3.5],
                   metavar=("Z", "Y", "X"), help="SPADE's native voxel spacing in mm")
    p.add_argument("--spacing", type=float, nargs=3, default=list(config.TARGET_SPACING_ZYX),
                   metavar=("Z", "Y", "X"), help="target voxel spacing in mm")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")

    orig_spacing = np.array(args.source_spacing, dtype=float)
    target_spacing = np.array(args.spacing, dtype=float)
    print(f"SPADE preprocessing\n  source: {args.source}\n  target: {args.target}\n"
          f"  spacing (z,y,x): {tuple(orig_spacing)} -> {tuple(target_spacing)}\n")

    n_ok = n_skip = n_err = 0
    # Layout is two levels deep: PatientID/ScanID/{pet,ct}.npy
    for pet_path in sorted(args.source.glob("*/*/pet.npy")):
        scan_dir = pet_path.parent
        ct_path = scan_dir / "ct.npy"
        if not ct_path.exists():
            n_skip += 1
            continue

        rel = pet_path.relative_to(args.source)
        save_path = args.target / rel
        if save_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            print(f"Processing {rel.parent}")
            pet = np.load(pet_path).astype(np.float32)
            ct = np.load(ct_path).astype(np.float32)

            pet_r = resample_to_spacing(pet, orig_spacing, target_spacing, order=1)
            ct_r = resample_to_shape(ct, pet_r.shape, order=1)

            bbox = get_body_bbox(
                ct_r, threshold=config.CT_BODY_THRESHOLD, pad=config.CROP_PAD, shape=pet_r.shape
            )
            if bbox is not None:
                pet_r, ct_r = pet_r[bbox], ct_r[bbox]

            stacked = np.stack([normalize_volume(pet_r), normalize_volume(ct_r)], axis=0)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, stacked)
            print(f"  -> {stacked.shape} -> {save_path}")
            n_ok += 1
        except Exception as e:
            print(f"  [error] {scan_dir}: {e}")
            n_err += 1

    print(f"\nDone. processed={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
