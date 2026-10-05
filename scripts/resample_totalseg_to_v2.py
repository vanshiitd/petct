#!/usr/bin/env python3
"""Move the TotalSegmentator organ masks onto the v2 grid.

The masks were produced from the old conversion, whose PET carried
`direction_y = +1`. v2's PET carries `direction_y = -1` with the origin moved to
the far edge, so the two describe the *same* physical volume traversed in
opposite directions. Physically the masks are already right; as numpy arrays
they are mirrored in y against everything in v2.

That is the mirrored-mask bug again, one level up: an organ-overlap rule built by
indexing these arrays against v2 arrays would mask the wrong side of every
patient, and nothing about the shapes or voxel counts would look wrong.

Resampling through physical space fixes it, and here it is exact rather than
approximate: the source and target grids have identical size, identical spacing
and identical extent, differing only in the direction they are traversed, so
nearest-neighbour resampling is a pure index flip. Every label keeps its voxel
count to the voxel -- which the script asserts per case rather than assuming.

This is used instead of re-running TotalSegmentator on v2's CTres, which would
need the GPU (busy) and many hours, to change nothing but the interpolation
kernel the CT was built with.

    python scripts/resample_totalseg_to_v2.py --totalseg D:\\data\\totalseg_ml \\
        --v2 C:\\data\\autopet_nifti_v2 --out D:\\data\\totalseg_ml_v2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def scan_dir(root: Path, case_id: str) -> Path:
    d = root / case_id
    if (d / "PET.nii.gz").exists():
        return d
    parts = case_id.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    return root / patient / case_id[len(patient) + 1:]


def extent(img) -> tuple:
    """The physical corners the image spans, independent of traversal direction."""
    size, spacing = img.GetSize(), img.GetSpacing()
    lo = img.TransformIndexToPhysicalPoint((0, 0, 0))
    hi = img.TransformIndexToPhysicalPoint(tuple(s - 1 for s in size))
    return (tuple(round(min(a, b), 3) for a, b in zip(lo, hi)),
            tuple(round(max(a, b), 3) for a, b in zip(lo, hi)),
            tuple(round(s, 6) for s in spacing), size)


def one(job) -> dict:
    case, ts_path, v2_root, out_path = job
    try:
        ts = sitk.ReadImage(str(ts_path))
        pet = sitk.ReadImage(str(scan_dir(Path(v2_root), case) / "PET.nii.gz"))

        if extent(ts) != extent(pet):
            return {"case": case, "status": "extent_differs",
                    "detail": f"{extent(ts)} vs {extent(pet)}"}

        out = sitk.Resample(ts, pet, sitk.Transform(), sitk.sitkNearestNeighbor,
                            0, ts.GetPixelID())

        a = sitk.GetArrayViewFromImage(ts)
        b = sitk.GetArrayViewFromImage(out)
        la, ca = np.unique(a, return_counts=True)
        lb, cb = np.unique(b, return_counts=True)
        if not (np.array_equal(la, lb) and np.array_equal(ca, cb)):
            return {"case": case, "status": "not_lossless",
                    "detail": f"{len(la)} labels in -> {len(lb)} out, counts differ"}

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(out_path), True)
        return {"case": case, "status": "ok", "labels": int(len(la)),
                "flipped": bool(not np.array_equal(a, b))}
    except Exception as e:
        return {"case": case, "status": "error", "detail": repr(e)[:200]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--totalseg", type=Path, required=True)
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    masks = sorted(args.totalseg.glob("*.nii.gz"))
    jobs = [(m.name[:-7], str(m), str(args.v2), str(args.out / m.name)) for m in masks]
    print(f"{len(jobs)} masks -> {args.out}\n")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(one, jobs, chunksize=2), 1):
            rows.append(r)
            if r["status"] != "ok":
                print(f"  {r['case']}: {r['status']} {r.get('detail', '')[:160]}")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\nresampled {len(ok)} | failed {len(bad)}")
    print(f"array actually changed on {sum(1 for r in ok if r['flipped'])} of {len(ok)} "
          f"(the rest already matched)")
    print("every resampled mask preserved its label voxel counts exactly"
          if ok and not bad else "CHECK THE FAILURES ABOVE")


if __name__ == "__main__":
    main()
