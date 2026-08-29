#!/usr/bin/env python3
"""Report the volume-dimension range of each preprocessed dataset.

Useful as a sanity check after preprocessing: a dataset whose minimum Z is
suspiciously small usually means a truncated or corrupt volume slipped through.

Example:
    python scripts/scan_dimensions.py --root /data/PETCTfoundation --exclude-anomalies
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=config.PATHS.pretrain_root,
                   help="root containing one subdirectory per dataset")
    p.add_argument("--exclude-anomalies", action="store_true",
                   help="skip the known-corrupt volumes listed in config.PRETRAIN_ANOMALIES")
    args = p.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Path does not exist: {args.root}")

    print(f"Scanning dataset dimensions in: {args.root}")
    if args.exclude_anomalies:
        print(f"Excluding {len(config.PRETRAIN_ANOMALIES)} known anomalous samples.")
    print("=" * 62)

    grand_total = 0
    for dataset_dir in sorted(args.root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        files = sorted(dataset_dir.rglob("*.npy"))
        if args.exclude_anomalies:
            files = [f for f in files
                     if not any(bad in f.as_posix() for bad in config.PRETRAIN_ANOMALIES)]
        if not files:
            continue

        mins = np.array([np.inf] * 3)
        maxs = np.array([0.0] * 3)
        n_valid = 0
        for f in files:
            try:
                # mmap: read the header only, never pull the array into memory
                shape = np.load(f, mmap_mode="r").shape
            except Exception as e:
                print(f"  [warn] unreadable {f.name}: {e}")
                continue
            zyx = np.array(shape[-3:], dtype=float)  # tolerate (2,Z,Y,X) or (Z,Y,X)
            mins = np.minimum(mins, zyx)
            maxs = np.maximum(maxs, zyx)
            n_valid += 1

        if n_valid == 0:
            continue
        grand_total += n_valid

        print(f"Dataset: {dataset_dir.name:<14}| Valid Tensors: {n_valid}")
        for axis, label in zip(range(3), ("Z-axis (Slice)", "Y-axis (H)    ", "X-axis (W)    ")):
            print(f"  -> {label}: Min = {int(mins[axis]):>4}, Max = {int(maxs[axis]):>4}")
        print("-" * 62)

    print(f"Total volumes: {grand_total}")


if __name__ == "__main__":
    main()
