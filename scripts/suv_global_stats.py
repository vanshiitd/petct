#!/usr/bin/env python3
"""Compute the ONE global SUV mean and std used to scale the PET channel.

Dataset503 z-scores PET per scan, which is the defect measured earlier: the same
tissue lands at a different value in every patient, so no fixed decision
boundary transfers. A single transform applied to every case keeps SUV's
calibration intact.

The statistics come from the TRAINING cases only. Val and test contribute
nothing, or the normalisation itself would carry information out of the held-out
sets.

Only voxels inside each case's body crop count, since that is exactly what gets
exported. Mean and std are accumulated as running sums, so the full 9 billion
voxels never have to be held at once.

    python scripts/suv_global_stats.py --data-root D:\\data\\autopet_nifti \\
        --split splits/autopet_v1.json \\
        --inverse D:\\data\\nnunet\\raw\\Dataset503_AutoPET_MAEprep\\inverse \\
        --out D:\\data\\reports\\suv_global_stats.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def accumulate(job) -> tuple[str, float, float, int, float, float]:
    """Returns (case, sum, sum_of_squares, count, min, max) over the crop."""
    case_id, scan_dir, meta_path = job
    meta = json.loads(Path(meta_path).read_text())
    suv = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(scan_dir) / "SUV.nii.gz"))).astype(np.float64)

    bbox = meta["crop_bbox_zyx"]
    if bbox is not None:
        (z0, z1), (y0, y1), (x0, x1) = bbox
        suv = suv[z0:z1, y0:y1, x0:x1]
    return (case_id, float(suv.sum()), float((suv ** 2).sum()), int(suv.size),
            float(suv.min()), float(suv.max()))


def scan_dir_for_case(data_root: Path, patient: str, case_id: str) -> Path:
    if case_id == patient:
        return data_root / patient
    return data_root / patient / case_id[len(patient) + 1:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--inverse", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    jobs = []
    for patient, case_ids in sorted(split["splits"]["train"]["cases"].items()):
        for case_id in case_ids:
            jobs.append((case_id,
                         str(scan_dir_for_case(args.data_root, patient, case_id)),
                         str(args.inverse / f"{case_id}.json")))
    print(f"accumulating SUV statistics over {len(jobs)} TRAINING cases "
          f"(val and test excluded)\n")

    total_sum = total_sq = 0.0
    total_n = 0
    vmin, vmax = np.inf, -np.inf
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (case, s, sq, n, lo, hi) in enumerate(pool.map(accumulate, jobs, chunksize=4), 1):
            total_sum += s
            total_sq += sq
            total_n += n
            vmin, vmax = min(vmin, lo), max(vmax, hi)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")

    mu = total_sum / total_n
    var = total_sq / total_n - mu ** 2
    sigma = float(np.sqrt(max(var, 0.0)))

    print(f"\nvoxels            : {total_n:,}")
    print(f"SUV mean (mu)     : {mu:.6f}")
    print(f"SUV std  (sigma)  : {sigma:.6f}")
    print(f"SUV range         : {vmin:.4f} .. {vmax:.1f}")
    print(f"\nthe PET channel becomes (SUV - {mu:.6f}) / {sigma:.6f}, identically for every case")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "mu": mu, "sigma": sigma,
        "n_voxels": total_n, "n_cases": len(jobs),
        "suv_min": vmin, "suv_max": vmax,
        "source": "training cases only, voxels inside the body crop",
        "split": str(args.split),
    }, indent=1))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
