#!/usr/bin/env python3
"""Precision, recall and lesion-level detection, to tell a real FP reduction apart
from a model that simply predicts less.

A lower false-positive volume is only good news if recall survives. A model that
shrinks every prediction lowers FP, raises FN, and can leave Dice unchanged --
which looks like an improvement in any report that quotes FP alone. These are
the numbers that separate the two cases:

  * **predicted volume per patient** -- is it just predicting less?
  * **voxel precision and recall**, pooled over voxels and averaged per case.
    Pooled is reported first because a per-case mean lets a patient with a
    4-voxel lesion count as much as one with 50,000.
  * **lesion-level detection** -- the fraction of ground-truth components with
    any predicted voxel on them. A model can lose a lot of voxel recall while
    still finding every lesion, and for a reading workflow that distinction is
    the whole point.
  * **false-positive lesions per patient** -- predicted components touching no
    ground truth, which is what a reader actually has to dismiss.

    python scripts/detection_metrics.py --pred <dir> --gt <dir> \\
        --out-csv reports/detect_505_10pct_test.csv
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def one(job) -> dict:
    case, pred_path, gt_path = job
    p_img = sitk.ReadImage(str(pred_path))
    g_img = sitk.ReadImage(str(gt_path))
    pred = sitk.GetArrayFromImage(p_img) > 0
    gt = sitk.GetArrayFromImage(g_img) > 0
    sx, sy, sz = p_img.GetSpacing()
    v = sx * sy * sz / 1000.0

    tp = int(np.logical_and(pred, gt).sum())
    n_pred, n_gt = int(pred.sum()), int(gt.sum())

    # lesion level
    gt_lab, gt_n = ndi.label(gt)
    hit = 0
    if gt_n:
        overlap = np.bincount(gt_lab[pred].ravel(), minlength=gt_n + 1)
        hit = int((overlap[1:] > 0).sum())
    p_lab, p_n = ndi.label(pred)
    fp_lesions = 0
    if p_n:
        ov = np.bincount(p_lab[gt].ravel(), minlength=p_n + 1)
        fp_lesions = int((ov[1:] == 0).sum())

    return {
        "case": case,
        "gt_positive": int(n_gt > 0),
        "pred_volume_ml": n_pred * v,
        "gt_volume_ml": n_gt * v,
        "tp_voxels": tp, "pred_voxels": n_pred, "gt_voxels": n_gt,
        "precision": (tp / n_pred) if n_pred else float("nan"),
        "recall": (tp / n_gt) if n_gt else float("nan"),
        "n_gt_lesions": gt_n, "n_gt_lesions_hit": hit,
        "lesion_detect": (hit / gt_n) if gt_n else float("nan"),
        "n_pred_lesions": p_n, "n_fp_lesions": fp_lesions,
    }


def summarise(rows: list[dict], label: str) -> dict:
    pos = [r for r in rows if r["gt_positive"] == 1]
    neg = [r for r in rows if r["gt_positive"] == 0]
    tp = sum(r["tp_voxels"] for r in pos)
    pv = sum(r["pred_voxels"] for r in rows)
    gv = sum(r["gt_voxels"] for r in pos)
    gl = sum(r["n_gt_lesions"] for r in pos)
    gh = sum(r["n_gt_lesions_hit"] for r in pos)
    out = {
        "pred_ml_all": float(np.mean([r["pred_volume_ml"] for r in rows])),
        "pred_ml_pos": float(np.mean([r["pred_volume_ml"] for r in pos])),
        "pred_ml_neg": float(np.mean([r["pred_volume_ml"] for r in neg])) if neg else float("nan"),
        "precision_pooled": tp / pv if pv else float("nan"),
        "recall_pooled": tp / gv if gv else float("nan"),
        "precision_mean": float(np.nanmean([r["precision"] for r in pos])),
        "recall_mean": float(np.nanmean([r["recall"] for r in pos])),
        "lesion_detect_pooled": gh / gl if gl else float("nan"),
        "lesion_detect_mean": float(np.mean([r["n_gt_lesions_hit"] / r["n_gt_lesions"]
                                             for r in pos if r["n_gt_lesions"]])),
        "fp_lesions_all": float(np.mean([r["n_fp_lesions"] for r in rows])),
        "fp_lesions_neg": float(np.mean([r["n_fp_lesions"] for r in neg])) if neg else float("nan"),
        "gt_lesions_total": gl,
    }
    print(f"\n### {label}  ({len(rows)} cases, {len(pos)} tumour-positive)")
    print(f"{'predicted volume mL, all patients':42s} {out['pred_ml_all']:>9.2f}")
    print(f"{'  tumour-positive patients':42s} {out['pred_ml_pos']:>9.2f}")
    print(f"{'  tumour-free patients':42s} {out['pred_ml_neg']:>9.2f}")
    print(f"{'voxel precision (pooled)':42s} {out['precision_pooled']:>9.4f}")
    print(f"{'voxel recall (pooled)':42s} {out['recall_pooled']:>9.4f}")
    print(f"{'voxel precision (per-case mean)':42s} {out['precision_mean']:>9.4f}")
    print(f"{'voxel recall (per-case mean)':42s} {out['recall_mean']:>9.4f}")
    print(f"{'lesions detected (pooled)':42s} {out['lesion_detect_pooled']:>9.4f}"
          f"   {gh}/{gl}")
    print(f"{'lesions detected (per-case mean)':42s} {out['lesion_detect_mean']:>9.4f}")
    print(f"{'false-positive lesions per patient':42s} {out['fp_lesions_all']:>9.2f}")
    print(f"{'  on tumour-free patients':42s} {out['fp_lesions_neg']:>9.2f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, action="append", required=True)
    p.add_argument("--label", action="append", default=None)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out-csv", type=Path, action="append", default=None)
    args = p.parse_args()

    labels = args.label or [d.name for d in args.pred]
    for i, pred_dir in enumerate(args.pred):
        files = sorted(pred_dir.glob("*.nii.gz"))
        jobs = [(f.name[:-7], str(f), str(args.gt / f.name)) for f in files]
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for r in pool.map(one, jobs, chunksize=2):
                rows.append(r)
        summarise(rows, labels[i])
        if args.out_csv and i < len(args.out_csv):
            args.out_csv[i].parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_csv[i], "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"   per-case CSV: {args.out_csv[i]}")


if __name__ == "__main__":
    main()
