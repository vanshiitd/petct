#!/usr/bin/env python3
"""Score predicted lesion masks against ground truth, in the original NIfTI grid.

The metrics the earlier runs reported were not comparable to anything:

  * MONAI's DiceMetric silently skips a case whose ground truth is empty, so the
    mean Dice was over tumour-positive cases only -- and roughly half of AutoPET
    is tumour-free, where a model can predict anything at all without it showing.
  * HD95 came out in voxels, which is not a distance unless every dataset shares
    one spacing.

So this reports, per case and in summary:

  * **Dice** over tumour-positive cases (the old convention, kept so numbers can
    still be compared) and, separately, how many tumour-free cases the model puts
    any foreground on at all.
  * **HD95 in millimetres**, using each image's own spacing, for cases where both
    masks are non-empty (it is undefined otherwise).
  * **False-positive and false-negative volume in mL**, autoPET-style: a predicted
    connected component that touches no true lesion is entirely false positive; a
    true lesion component that the prediction never touches is entirely missed.
    This is what actually matters clinically and it is defined for every case,
    including the tumour-free half.

    python scripts/evaluate_predictions.py --pred <dir> --gt <labelsTs> \\
        --out-csv reports/eval_nnunet.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Binary mask as (z, y, x) plus voxel spacing in mm, also (z, y, x)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    sx, sy, sz = img.GetSpacing()          # SimpleITK reports (x, y, z)
    return arr, (sz, sy, sx)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")                # both empty: undefined, not 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Voxels on the mask boundary (mask minus its erosion)."""
    if not mask.any():
        return np.empty((0, 3), dtype=np.int64)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return np.argwhere(mask & ~eroded)


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """95th-percentile symmetric Hausdorff distance, in millimetres.

    Distances come from a spacing-aware EDT, so the result is a real distance
    rather than a voxel count -- comparable across datasets and resolutions.
    """
    if not pred.any() or not gt.any():
        return float("nan")

    # distance from every voxel to the nearest surface voxel of the other mask
    dt_to_gt = ndi.distance_transform_edt(~gt, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred, sampling=spacing)

    pred_surf = surface_voxels(pred)
    gt_surf = surface_voxels(gt)
    d_pred_gt = dt_to_gt[tuple(pred_surf.T)]
    d_gt_pred = dt_to_pred[tuple(gt_surf.T)]
    both = np.concatenate([d_pred_gt, d_gt_pred])
    return float(np.percentile(both, 95))


def fp_fn_volumes_ml(pred: np.ndarray, gt: np.ndarray,
                     spacing: tuple[float, float, float]) -> tuple[float, float, int, int]:
    """autoPET-style false-positive / false-negative volumes, in mL.

    A predicted component that overlaps no true lesion is a wholly spurious
    finding, and a true lesion the prediction never touches is wholly missed.
    Scoring whole components rather than stray voxels is what makes this
    clinically meaningful: shaving a few voxels off a correctly-found lesion is
    not the same kind of error as inventing a lesion that is not there.
    """
    voxel_ml = float(np.prod(spacing)) / 1000.0     # mm^3 -> mL

    fp_vol = 0.0
    n_fp = 0
    if pred.any():
        lab, n = ndi.label(pred)
        for i in range(1, n + 1):
            comp = lab == i
            if not np.logical_and(comp, gt).any():
                fp_vol += comp.sum() * voxel_ml
                n_fp += 1

    fn_vol = 0.0
    n_fn = 0
    if gt.any():
        lab, n = ndi.label(gt)
        for i in range(1, n + 1):
            comp = lab == i
            if not np.logical_and(comp, pred).any():
                fn_vol += comp.sum() * voxel_ml
                n_fn += 1

    return fp_vol, fn_vol, n_fp, n_fn


def evaluate_case(pred_path: Path, gt_path: Path) -> dict:
    pred, spacing = load_mask(pred_path)
    gt, gt_spacing = load_mask(gt_path)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if not np.allclose(spacing, gt_spacing, atol=1e-3):
        raise ValueError(f"spacing mismatch: pred {spacing} vs gt {gt_spacing}")

    voxel_ml = float(np.prod(spacing)) / 1000.0
    fp_vol, fn_vol, n_fp, n_fn = fp_fn_volumes_ml(pred, gt, spacing)
    return {
        "case": pred_path.name.replace(".nii.gz", ""),
        "gt_positive": int(gt.any()),
        "pred_positive": int(pred.any()),
        "dice": dice(pred, gt),
        "hd95_mm": hd95_mm(pred, gt, spacing),
        "gt_volume_ml": gt.sum() * voxel_ml,
        "pred_volume_ml": pred.sum() * voxel_ml,
        "fp_volume_ml": fp_vol,
        "fn_volume_ml": fn_vol,
        "n_fp_components": n_fp,
        "n_fn_components": n_fn,
        "spacing_zyx": "x".join(f"{s:.3f}" for s in spacing),
    }


def summarise(rows: list[dict]) -> None:
    pos = [r for r in rows if r["gt_positive"]]
    neg = [r for r in rows if not r["gt_positive"]]
    dices = [r["dice"] for r in pos if r["dice"] == r["dice"]]
    hd = [r["hd95_mm"] for r in rows if r["hd95_mm"] == r["hd95_mm"]]
    fp_on_neg = [r for r in neg if r["pred_positive"]]
    # A tumour-positive case the model predicts nothing on has no HD95 (the
    # distance to an empty set is undefined), so it silently leaves the HD95
    # mean -- exactly the cases a model does worst on. Report how many there
    # are and what fraction of the positives the mean actually covers.
    missed = [r for r in pos if not r["pred_positive"]]

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    print("\n" + "=" * 62)
    print(f"{'cases':38s} {len(rows):>10d}")
    print(f"{'  tumour-positive':38s} {len(pos):>10d}")
    print(f"{'  tumour-free':38s} {len(neg):>10d}")
    print("-" * 62)
    print(f"{'Dice (tumour-positive cases)':38s} {mean(dices):>10.4f}")
    print(f"{'  median':38s} {float(np.median(dices)) if dices else float('nan'):>10.4f}")
    print(f"{'  cases scored':38s} {len(dices):>10d}")
    print(f"{'  complete misses (empty prediction)':38s} {len(missed):>10d}"
          f"  ({len(missed)/max(1,len(pos)):.1%} of positives, Dice 0)")
    print(f"{'HD95 mm (both masks non-empty)':38s} {mean(hd):>10.4f}")
    print(f"{'  median':38s} {float(np.median(hd)) if hd else float('nan'):>10.4f}")
    print(f"{'  cases scored':38s} {len(hd):>10d}"
          f"  of {len(pos)} positives ({len(hd)/max(1,len(pos)):.1%} covered)")
    if missed:
        print(f"{'  NOT covered by the HD95 mean':38s} {len(missed):>10d}"
              f"  (undefined on an empty prediction)")
    print("-" * 62)
    print(f"{'tumour-free cases with any FP':38s} {len(fp_on_neg):>10d}"
          f"  ({len(fp_on_neg)/max(1,len(neg)):.1%})")
    print(f"{'mean FP volume, mL (all cases)':38s} {mean([r['fp_volume_ml'] for r in rows]):>10.4f}")
    print(f"{'mean FN volume, mL (all cases)':38s} {mean([r['fn_volume_ml'] for r in rows]):>10.4f}")
    print(f"{'mean FP volume on tumour-free, mL':38s} {mean([r['fp_volume_ml'] for r in neg]):>10.4f}")
    print("=" * 62)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True, help="directory of predicted masks")
    p.add_argument("--gt", type=Path, required=True, help="directory of ground-truth masks")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--suffix", default=".nii.gz")
    args = p.parse_args()

    preds = sorted(args.pred.glob(f"*{args.suffix}"))
    if not preds:
        raise SystemExit(f"No *{args.suffix} under {args.pred}")

    rows, missing = [], []
    for i, pred_path in enumerate(preds, 1):
        gt_path = args.gt / pred_path.name
        if not gt_path.exists():
            missing.append(pred_path.name)
            continue
        rows.append(evaluate_case(pred_path, gt_path))
        if i % 25 == 0:
            print(f"  {i}/{len(preds)}")

    if missing:
        print(f"WARNING: {len(missing)} prediction(s) had no ground truth: {missing[:5]}")
    if not rows:
        raise SystemExit("No case could be evaluated.")

    summarise(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-case CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
