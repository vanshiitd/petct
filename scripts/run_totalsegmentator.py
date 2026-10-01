#!/usr/bin/env python3
"""Segment organs on CT_resample.nii.gz for a list of cases, in one process.

Why the Python API rather than the CLI: invoking TotalSegmentator once per case
spends most of its time on startup and model loading, not on the scan. Keeping
one process alive across hundreds of cases amortises that away. Multilabel
output (`ml`) writes a single file instead of one per organ, which also removes
most of the saving cost.

CT_resample.nii.gz is already on the PET grid, so these masks line up voxel for
voxel with predictions mapped back to the original scan -- no resampling needed
when they are used.

Fast mode (3 mm) is enough here: the organs in question are large, and the masks
are only used to ask whether a predicted blob sits inside one.

    python scripts/run_totalsegmentator.py --cases cases.txt \\
        --data-root D:\\data\\autopet_nifti --out D:\\data\\totalseg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# the organs considered for masking, plus two kept only as a reference point
ROI_SUBSET = ["brain", "urinary_bladder", "kidney_left", "kidney_right",
              "heart", "liver", "spleen"]


def scan_dir_for_case(data_root: Path, case_id: str) -> Path:
    """Case ids are <patient> or <patient>_<study>; the study part is a subdir."""
    direct = data_root / case_id
    if (direct / "CT_resample.nii.gz").exists():
        return direct
    patient = case_id.split("_")[0] + "_" + case_id.split("_")[1]
    nested = data_root / patient / case_id[len(patient) + 1:]
    return nested


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, required=True, help="file with one case id per line")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument("--full-res", dest="fast", action="store_false")
    args = p.parse_args()

    from totalsegmentator.python_api import totalsegmentator

    cases = [c.strip() for c in args.cases.read_text().splitlines() if c.strip()]
    print(f"{len(cases)} cases | fast={args.fast} | roi_subset={ROI_SUBSET}\n", flush=True)

    times, done, skipped, failed = [], 0, 0, []
    for i, case in enumerate(cases, 1):
        out_file = args.out / f"{case}.nii.gz"
        if out_file.exists():
            skipped += 1
            continue
        scan = scan_dir_for_case(args.data_root, case) / "CT_resample.nii.gz"
        if not scan.exists():
            failed.append((case, "no CT_resample.nii.gz"))
            continue
        t0 = time.time()
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            totalsegmentator(str(scan), str(out_file), ml=True, fast=args.fast,
                             roi_subset=ROI_SUBSET, device="gpu", quiet=True,
                             nr_thr_saving=6)
            times.append(time.time() - t0)
            done += 1
        except Exception as e:
            failed.append((case, repr(e)[:200]))
        if i % 10 == 0 or i == len(cases):
            mean = sum(times) / len(times) if times else 0
            left = (len(cases) - i) * mean / 60
            print(f"  {i}/{len(cases)} | done {done} skipped {skipped} failed {len(failed)} "
                  f"| {mean:.1f}s/case | ~{left:.0f} min left", flush=True)

    mean = sum(times) / len(times) if times else 0
    print(f"\nsegmented {done} | already present {skipped} | failed {len(failed)}")
    print(f"mean {mean:.1f}s per case")
    for c, e in failed[:10]:
        print(f"   {c}: {e}")
    (args.out / "_run_info.json").write_text(json.dumps({
        "n_done": done, "n_skipped": skipped, "n_failed": len(failed),
        "mean_seconds_per_case": mean, "fast": args.fast, "roi_subset": ROI_SUBSET,
        "failed": failed,
    }, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
