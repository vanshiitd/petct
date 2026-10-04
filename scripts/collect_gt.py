#!/usr/bin/env python3
"""Gather original-space ground-truth masks into the flat layout the evaluator wants.

`evaluate_predictions.py` looks the ground truth up as `--gt / <case>.nii.gz`,
but the collection stores it nested, one directory per scan, and repeat-visit
patients nest a second level. This walks the frozen split and copies each
requested set's `tumorSeg.nii.gz` out under its case id.

Copies rather than links: the masks live on a different volume from the reports,
so a hard link is not available, and they are small (a few hundred KB gzipped,
almost entirely zeros).

    python scripts/collect_gt.py --root C:\\data\\autopet_nifti_v2 \\
        --split splits/autopet_v1.json --set test --out D:\\data\\nnunet\\gt_v2_test
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import SimpleITK as sitk


def scan_dir(root: Path, case_id: str) -> Path:
    """Where a case lives: flat for single-visit patients, nested for repeats."""
    d = root / case_id
    if (d / "tumorSeg.nii.gz").exists():
        return d
    parts = case_id.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    return root / patient / case_id[len(patient) + 1:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="converted dataset root")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--set", dest="which", required=True,
                   choices=["train", "val", "test"])
    p.add_argument("--cases", type=Path, default=None,
                   help="restrict to the case ids in this file, one per line")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    cases = [c for cs in split["splits"][args.which]["cases"].values() for c in cs]
    if args.cases:
        wanted = {ln.strip() for ln in args.cases.read_text().splitlines() if ln.strip()}
        unknown = wanted - set(cases)
        if unknown:
            raise SystemExit(f"{len(unknown)} case(s) not in '{args.which}': "
                             f"{sorted(unknown)[:5]}")
        cases = [c for c in cases if c in wanted]

    args.out.mkdir(parents=True, exist_ok=True)
    n_pos = 0
    for case in cases:
        src = scan_dir(args.root, case) / "tumorSeg.nii.gz"
        if not src.exists():
            raise SystemExit(f"missing ground truth for {case}: {src}")
        shutil.copy(src, args.out / f"{case}.nii.gz")
        n_pos += int(sitk.GetArrayFromImage(sitk.ReadImage(str(src))).any())

    print(f"{args.which}: {len(cases)} masks -> {args.out}")
    print(f"  tumour-positive: {n_pos}")
    expected = split["splits"][args.which]["n_tumour_positive_scans"]
    if args.cases is None:
        print(f"  split file says: {expected}  "
              f"{'OK' if n_pos == expected else 'MISMATCH'}")


if __name__ == "__main__":
    main()
