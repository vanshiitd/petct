#!/usr/bin/env python3
"""Audit R9(1): re-measure the mask-flip test on the corrected v2 conversion.

`check_mask_flip.py` ran on 2026-10-02 against the original conversion and is
the source of the pre-fix figures (308 of 501 flip-better, median uptake ratio
2.57x stored against 4.30x flipped). The post-fix count quoted as "flipping
improves uptake in only 4 of 501" has no saved log, so this recomputes it.

Same test as the original - median PET uptake inside the mask against inside the
left-right flipped mask, both over the body median - but reading the **v2**
conversion, and writing to a new file. `check_mask_flip.py` hardcodes its output
path, so running it would overwrite the pre-fix record; this does not touch it.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

V2 = Path(r"C:\data\autopet_nifti_v2")


def dirs_for(case: str) -> Path:
    d = V2 / case
    if (d / "PET.nii.gz").exists():
        return d
    parts = case.split("_")
    p = f"{parts[0]}_{parts[1]}"
    return V2 / p / case[len(p) + 1:]


def one(case: str):
    d = dirs_for(case)
    try:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "tumorSeg.nii.gz"))) > 0
        if not seg.any():
            return None
        pet = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "PET.nii.gz")))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "CT_resample.nii.gz")))
        body = ct > -500
        bg = float(np.median(pet[body & ~seg])) if (body & ~seg).any() else 1.0
        bg = max(bg, 1.0)
        as_is = float(np.median(pet[seg])) / bg
        flipped = seg[:, ::-1, :]
        as_flipped = float(np.median(pet[flipped])) / bg
        return {"case": case, "ratio_as_is": as_is, "ratio_flipped": as_flipped,
                "flip_better": bool(as_flipped > as_is * 1.2),
                "voxels": int(seg.sum())}
    except Exception as e:                      # noqa: BLE001
        return {"case": case, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_flip_recheck.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    cases = sorted({p.parent.name if p.parent.parent == V2 else
                    f"{p.parent.parent.name}_{p.parent.name}"
                    for p in V2.rglob("tumorSeg.nii.gz")})
    print(f"{len(cases)} scans with a tumour mask in {V2}")

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, cases), 1):
            if r is not None:
                rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(cases)}", flush=True)

    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    fb = [r for r in ok if r["flip_better"]]
    ra = np.array([r["ratio_as_is"] for r in ok])
    rf = np.array([r["ratio_flipped"] for r in ok])
    print(f"\nscans measured: {len(ok)}  ({len(bad)} errors)")
    print(f"mask sits on BRIGHTER tissue when flipped : {len(fb)} "
          f"({100*len(fb)/max(len(ok),1):.1f}%)")
    print(f"uptake ratio as stored : median {np.median(ra):.2f}")
    print(f"uptake ratio flipped   : median {np.median(rf):.2f}")
    args.out.write_text(json.dumps(
        {"n": len(ok), "n_errors": len(bad), "flip_better": len(fb),
         "median_ratio_as_stored": float(np.median(ra)),
         "median_ratio_flipped": float(np.median(rf)),
         "rows": rows}, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
