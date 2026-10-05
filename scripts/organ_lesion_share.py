#!/usr/bin/env python3
"""How much REAL lesion volume sits inside each organ?

Organ masking deletes predictions that fall inside an organ. That is only safe
for organs where real lesions essentially never occur. Brain metastases do
happen in melanoma, and this dataset is largely melanoma, lymphoma and lung
cancer -- so which organs are safe has to be measured on the ground truth, not
assumed from physiology.

Measured on TRAINING cases only, so the decision does not look at val or test.

Reports, per organ: the share of all lesion volume inside it, how many scans
have any lesion volume there, and the worst single scan. An organ may be used
for masking only if its share is negligible (the task sets < 0.5%).

    python scripts/organ_lesion_share.py --cases train_cases.txt \\
        --data-root D:\\data\\autopet_nifti --totalseg D:\\data\\totalseg_ml
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def scan_dir_for_case(data_root: Path, case_id: str) -> Path:
    direct = data_root / case_id
    if (direct / "tumorSeg.nii.gz").exists():
        return direct
    parts = case_id.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    return data_root / patient / case_id[len(patient) + 1:]


def one(job) -> dict | None:
    case, data_root, totalseg, label_ids = job
    ts_path = Path(totalseg) / f"{case}.nii.gz"
    if not ts_path.exists():
        return None
    seg_path = scan_dir_for_case(Path(data_root), case) / "tumorSeg.nii.gz"
    gt_img = sitk.ReadImage(str(seg_path))
    gt = sitk.GetArrayFromImage(gt_img) > 0
    if not gt.any():
        return {"case": case, "lesion_voxels": 0, "per_organ": {}}

    ts_img = sitk.ReadImage(str(ts_path))
    ts = sitk.GetArrayFromImage(ts_img)
    # Geometry, not just shape. Organ masks built from the old conversion have
    # the same (400, 400, N) shape as v2's labels but traverse y in the opposite
    # direction, so indexing one against the other mirrors every organ while
    # every shape check passes. That is the bug this whole reconversion exists
    # to fix; do not let it back in one level up.
    def geom(img):
        return (img.GetSize(),
                tuple(round(v, 6) for v in img.GetSpacing()),
                tuple(round(v, 3) for v in img.GetOrigin()),
                tuple(round(v, 6) for v in img.GetDirection()))

    if geom(ts_img) != geom(gt_img):
        return {"case": case,
                "error": f"organ mask geometry {geom(ts_img)} != label {geom(gt_img)}"}
    sx, sy, sz = ts_img.GetSpacing()
    voxel_ml = sx * sy * sz / 1000.0

    total = int(gt.sum())
    inside = np.bincount(ts[gt].ravel(), minlength=max(label_ids.values()) + 1)
    return {
        "case": case,
        "lesion_voxels": total,
        "lesion_ml": total * voxel_ml,
        "voxel_ml": voxel_ml,
        "per_organ": {name: int(inside[i]) if i < len(inside) else 0
                      for name, i in label_ids.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--totalseg", type=Path, required=True)
    p.add_argument("--label-ids", type=Path,
                   default=Path(r"D:\data\reports\totalseg_label_ids.json"))
    p.add_argument("--out", type=Path, default=Path(r"D:\data\reports\organ_lesion_share.json"))
    p.add_argument("--threshold", type=float, default=0.5,
                   help="percent of lesion volume above which an organ is NOT safe to mask")
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()

    label_ids = {k: int(v) for k, v in json.loads(args.label_ids.read_text()).items()}
    cases = [c.strip() for c in args.cases.read_text().splitlines() if c.strip()]
    jobs = [(c, str(args.data_root), str(args.totalseg), label_ids) for c in cases]

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(one, jobs, chunksize=4), 1):
            if r is not None:
                rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    pos = [r for r in rows if r["lesion_voxels"] > 0]
    if errs:
        print(f"\n{len(errs)} case(s) with mismatched masks:")
        for r in errs[:5]:
            print(f"   {r['case']}: {r['error']}")

    total_lesion = sum(r["lesion_voxels"] for r in pos)
    print(f"\ncases with organ masks: {len(rows)} | tumour-positive: {len(pos)}")
    print(f"total lesion volume: {total_lesion:,} voxels\n")

    print(f"{'organ':18s} {'share of lesion':>16s} {'scans affected':>15s} "
          f"{'worst scan':>12s}  verdict")
    print("-" * 78)
    summary = {}
    for name in label_ids:
        inside = sum(r["per_organ"].get(name, 0) for r in pos)
        share = inside / total_lesion if total_lesion else 0.0
        affected = sum(1 for r in pos if r["per_organ"].get(name, 0) > 0)
        worst = max((r["per_organ"].get(name, 0) / r["lesion_voxels"]
                     for r in pos if r["lesion_voxels"]), default=0.0)
        safe = share * 100 < args.threshold
        summary[name] = {"share_pct": share * 100, "scans_affected": affected,
                         "worst_scan_pct": worst * 100, "safe_to_mask": bool(safe),
                         "lesion_voxels_inside": inside}
        print(f"{name:18s} {share*100:15.4f}% {affected:15d} {worst*100:11.1f}%  "
              f"{'SAFE' if safe else 'NOT SAFE'}")

    allowed = [n for n, s in summary.items() if s["safe_to_mask"]]
    print(f"\norgans below {args.threshold}% of lesion volume (usable for masking):")
    print(f"  {allowed}")
    print(f"excluded: {[n for n in summary if n not in allowed]}")

    args.out.write_text(json.dumps({
        "threshold_pct": args.threshold, "n_cases": len(rows), "n_positive": len(pos),
        "total_lesion_voxels": total_lesion, "per_organ": summary,
        "allowed_for_masking": allowed,
    }, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
