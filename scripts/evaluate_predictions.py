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
from scipy import stats


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


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union. Undefined when both masks are empty, like Dice."""
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Voxels on the mask boundary (mask minus its erosion)."""
    if not mask.any():
        return np.empty((0, 3), dtype=np.int64)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return np.argwhere(mask & ~eroded)


def surface_distances_mm(pred: np.ndarray, gt: np.ndarray,
                         spacing: tuple[float, float, float]):
    """Directed surface distances in mm, or None if either mask is empty.

    Returns (pred surface -> gt, gt surface -> pred). Both HD95 and ASSD are
    derived from these, so the two distance-based metrics always describe the
    same geometry and the four expensive EDTs collapse to two.

    Convention, which matters when comparing to published numbers: distance is
    measured to the other mask's *volume*, so a predicted surface voxel lying
    inside the ground truth scores 0. medpy measures to the other mask's
    *border* instead, which is never smaller. This file has always used the
    volume convention for HD95 and keeps it, so the whole project's HD95 values
    remain comparable with each other; ASSD follows the same convention rather
    than mixing two distance fields in one row. Both therefore read slightly
    lower than medpy would report, most noticeably for a prediction contained
    entirely inside the truth.
    """
    if not pred.any() or not gt.any():
        return None

    # distance from every voxel to the nearest voxel of the other mask
    dt_to_gt = ndi.distance_transform_edt(~gt, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred, sampling=spacing)

    pred_surf = surface_voxels(pred)
    gt_surf = surface_voxels(gt)
    return dt_to_gt[tuple(pred_surf.T)], dt_to_pred[tuple(gt_surf.T)]


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
            dists=None) -> float:
    """95th-percentile symmetric Hausdorff distance, in millimetres.

    Distances come from a spacing-aware EDT, so the result is a real distance
    rather than a voxel count -- comparable across datasets and resolutions.
    """
    d = surface_distances_mm(pred, gt, spacing) if dists is None else dists
    if d is None:
        return float("nan")
    return float(np.percentile(np.concatenate(d), 95))


def assd_mm(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
            dists=None) -> float:
    """Average symmetric surface distance, in millimetres.

    The mean of the two directed average distances, which is medpy's definition
    of ASSD -- not the mean over the pooled distances, which weights whichever
    mask has more surface voxels. Where HD95 reports the worst disagreement,
    this reports the typical one, so a prediction that is right almost
    everywhere but badly wrong in one place separates the two.
    """
    d = surface_distances_mm(pred, gt, spacing) if dists is None else dists
    if d is None:
        return float("nan")
    d_pred_gt, d_gt_pred = d
    return float((d_pred_gt.mean() + d_gt_pred.mean()) / 2.0)


def component_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """How many connected components each mask has."""
    n_pred = int(ndi.label(pred)[1]) if pred.any() else 0
    n_gt = int(ndi.label(gt)[1]) if gt.any() else 0
    return n_pred, n_gt


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

    def unmatched(mask: np.ndarray, other: np.ndarray) -> tuple[float, int]:
        """Volume and count of `mask`'s components that touch no part of `other`.

        Labelled once, then both the component sizes and their overlaps come out
        of two bincounts. Masking each component separately would mean a
        full-volume pass per component, which is minutes per case once a model
        over-predicts into dozens of components.
        """
        if not mask.any():
            return 0.0, 0
        lab, n = ndi.label(mask)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        overlap = np.bincount(lab[other].ravel(), minlength=n + 1)
        unmatched_labels = np.nonzero(overlap[1:] == 0)[0] + 1   # label 0 is background
        return float(sizes[unmatched_labels].sum()) * voxel_ml, int(unmatched_labels.size)

    fp_vol, n_fp = unmatched(pred, gt)
    fn_vol, n_fn = unmatched(gt, pred)
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
    dists = surface_distances_mm(pred, gt, spacing)      # computed once, used twice
    n_pred_cc, n_gt_cc = component_counts(pred, gt)
    gt_ml = gt.sum() * voxel_ml
    pred_ml = pred.sum() * voxel_ml
    hit = n_gt_cc - n_fn                                  # true lesions the prediction touches
    return {
        "case": pred_path.name.replace(".nii.gz", ""),
        "gt_positive": int(gt.any()),
        "pred_positive": int(pred.any()),
        "dice": dice(pred, gt),
        "hd95_mm": hd95_mm(pred, gt, spacing, dists),
        "gt_volume_ml": gt_ml,
        "pred_volume_ml": pred_ml,
        "fp_volume_ml": fp_vol,
        "fn_volume_ml": fn_vol,
        "n_fp_components": n_fp,
        "n_fn_components": n_fn,
        "spacing_zyx": "x".join(f"{s:.3f}" for s in spacing),
        # --- added for Experiment 1 parity ---------------------------------
        "iou": iou(pred, gt),
        "assd_mm": assd_mm(pred, gt, spacing, dists),
        "volume_error_ml": pred_ml - gt_ml,
        "abs_volume_error_ml": abs(pred_ml - gt_ml),
        "rel_volume_error": ((pred_ml - gt_ml) / gt_ml) if gt_ml > 0 else float("nan"),
        "n_pred_components": n_pred_cc,
        "n_gt_components": n_gt_cc,
        "n_gt_lesions_hit": hit,
        "lesion_sensitivity": (hit / n_gt_cc) if n_gt_cc else float("nan"),
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

    # --- Experiment 1 parity metrics -------------------------------------
    ious = [r["iou"] for r in pos if r["iou"] == r["iou"]]
    assd = [r["assd_mm"] for r in rows if r["assd_mm"] == r["assd_mm"]]
    verr = [r["volume_error_ml"] for r in pos]
    aerr = [r["abs_volume_error_ml"] for r in pos]
    rerr = [r["rel_volume_error"] for r in pos if r["rel_volume_error"] == r["rel_volume_error"]]
    sens = [r["lesion_sensitivity"] for r in pos if r["lesion_sensitivity"] == r["lesion_sensitivity"]]
    gt_cc = sum(r["n_gt_components"] for r in pos)
    hit_cc = sum(r["n_gt_lesions_hit"] for r in pos)

    print("-" * 62)
    print(f"{'IoU (tumour-positive cases)':38s} {mean(ious):>10.4f}")
    print(f"{'  median':38s} {float(np.median(ious)) if ious else float('nan'):>10.4f}")
    print(f"{'ASSD mm (both masks non-empty)':38s} {mean(assd):>10.4f}")
    print(f"{'  median':38s} {float(np.median(assd)) if assd else float('nan'):>10.4f}")
    print(f"{'  cases scored':38s} {len(assd):>10d}")
    print("-" * 62)
    print(f"{'volume error mL (pred - true, pos)':38s} {mean(verr):>+10.4f}")
    print(f"{'  absolute':38s} {mean(aerr):>10.4f}")
    # Median first, because the mean of a ratio is not a summary of this
    # quantity: a 30 mL over-prediction on a 0.4 mL lesion is +8000%, so a
    # couple of sub-millilitre cases can set the mean while saying nothing
    # about how the model behaves on the lesions that carry the Dice.
    print(f"{'  relative, median':38s} "
          f"{float(np.median(rerr)) if rerr else float('nan'):>+10.2%}")
    if rerr:
        q1, q3 = np.percentile(rerr, [25, 75])
        print(f"{'    interquartile range':38s} {q1:>+9.1%} to {q3:+.1%}")
    print(f"{'  relative, mean (small-lesion heavy)':38s} {mean(rerr):>+10.2%}")

    # Correlation of predicted against true total tumour volume. Reported over
    # tumour-positive cases and over all cases: including the tumour-free half
    # pins a large cluster at true volume 0, which inflates r on its own, so the
    # two numbers answer different questions and both are given.
    def corr(subset):
        if len(subset) < 3:
            return float("nan"), float("nan")
        a = np.array([r["gt_volume_ml"] for r in subset])
        b = np.array([r["pred_volume_ml"] for r in subset])
        if a.std() == 0 or b.std() == 0:
            return float("nan"), float("nan")
        pear = float(np.corrcoef(a, b)[0, 1])
        # scipy's spearmanr, not argsort ranks: the tumour-free half all have a
        # true volume of exactly 0, and argsort breaks that tie arbitrarily,
        # which silently invents an ordering inside the largest group in the
        # data. Average ranks are the only correct handling here.
        spear = float(stats.spearmanr(a, b).statistic)
        return pear, spear

    pp, ps = corr(pos)
    ap, asp = corr(rows)
    print(f"{'volume corr, tumour-positive (r)':38s} {pp:>10.4f}")
    print(f"{'  Spearman':38s} {ps:>10.4f}")
    print(f"{'volume corr, all cases (r)':38s} {ap:>10.4f}")
    print(f"{'  Spearman':38s} {asp:>10.4f}")
    print("-" * 62)
    print(f"{'predicted lesions per case':38s} "
          f"{mean([r['n_pred_components'] for r in rows]):>10.2f}")
    print(f"{'true lesions per positive case':38s} "
          f"{mean([r['n_gt_components'] for r in pos]):>10.2f}")
    print(f"{'lesion sensitivity (pooled)':38s} {hit_cc / max(1, gt_cc):>10.4f}"
          f"   {hit_cc}/{gt_cc}")
    print(f"{'  per-case mean':38s} {mean(sens):>10.4f}")
    print(f"{'false-positive lesions per case':38s} "
          f"{mean([r['n_fp_components'] for r in rows]):>10.2f}")
    print(f"{'  on tumour-free cases':38s} "
          f"{mean([r['n_fp_components'] for r in neg]):>10.2f}")
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
