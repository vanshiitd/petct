#!/usr/bin/env python3
"""Two worked examples for the Phase III report: a removal and a recovery.

Coronal maximum-intensity projections, as in Phase II -- a single slice hides
findings that lie off it, which is exactly what these figures exist to show.
Each row is PET with M0, PET with M1, PET with the ground truth, and the
detector's view (P(FP) painted on the components it judged).

Cases are chosen from the results rather than picked by eye: the removal
example is the tumour-free case where the refinement deleted the most
false-positive volume, and the recovery example is the case where it added a
component that genuinely overlaps a lesion M0 had missed.

    python scripts/exp4_figures.py --variant ABcons --out D:\\data\\reports
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import SimpleITK as sitk              # noqa: E402
from scipy import ndimage as ndi      # noqa: E402

REP = Path(r"D:\data\reports")
PRED = Path(r"D:\data\nnunet\predictions")
GT = Path(r"D:\data\nnunet\gt_v2_test")
INV = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2\inverse")


def mip(a, axis=1):
    return a.max(axis=axis)


def pick(variant: str):
    with open(REP / "evalM0_test.csv", newline="", encoding="utf-8") as f:
        m0 = {r["case"]: r for r in csv.DictReader(f)}
    with open(REP / f"eval_{variant}.csv", newline="", encoding="utf-8") as f:
        m1 = {r["case"]: r for r in csv.DictReader(f)}
    fnum = lambda v: float(v) if v not in ("", "nan") else float("nan")
    # removal: tumour-free case losing the most false-positive volume
    negs = [(fnum(m0[c]["fp_volume_ml"]) - fnum(m1[c]["fp_volume_ml"]), c)
            for c in m0 if m0[c]["gt_positive"] == "0"]
    rem = max(negs)[1]
    # recovery: a case whose GT lesion count hit by M1 exceeds M0's
    best, bestgain = None, 0
    for c in m0:
        if m0[c]["gt_positive"] != "1":
            continue
        gain = int(m1[c]["n_gt_lesions_hit"]) - int(m0[c]["n_gt_lesions_hit"])
        if gain > bestgain:
            best, bestgain = c, gain
    return rem, best, bestgain


def panel(ax, pet, overlay, cmap, title, vmax):
    ax.imshow(pet, cmap="gray_r", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    if overlay is not None and overlay.any():
        m = np.ma.masked_where(~overlay, overlay.astype(float))
        ax.imshow(m, cmap=cmap, alpha=0.7, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="ABcons",
                    help="variant for the removal example")
    ap.add_argument("--recovery-variant", default=None,
                    help="variant for the recovery example; the conservative "
                         "point recovers nothing, so the two examples may need "
                         "to come from different operating points")
    ap.add_argument("--out", type=Path, default=REP)
    args = ap.parse_args()

    rem, _, _ = pick(args.variant)
    rec_variant = args.recovery_variant or args.variant
    _, rec, gain = pick(rec_variant)
    print(f"removal example : {rem}")
    print(f"recovery example: {rec} (+{gain} ground-truth lesion(s) hit)")
    picks = [(rem, "false positives removed, " + args.variant),
             (rec, f"lesion recovered (+{gain}), " + rec_variant)]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4))
    for row, (case, caption) in enumerate(picks):
        if case is None:
            for c in range(3):
                axes[row, c].axis("off")
            continue
        meta = json.loads((INV / f"{case}.json").read_text())
        suv = sitk.GetArrayFromImage(sitk.ReadImage(meta["source_pet"])).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(GT / f"{case}.nii.gz"))) > 0
        a = sitk.GetArrayFromImage(sitk.ReadImage(
            str(PRED / "test_M0_mapped" / f"{case}.nii.gz"))) > 0
        vv = args.variant if row == 0 else rec_variant
        b = sitk.GetArrayFromImage(sitk.ReadImage(
            str(PRED / f"test_{vv}_mapped" / f"{case}.nii.gz"))) > 0
        pet2 = mip(suv)
        vmax = float(np.percentile(suv, 99.5)) or 1.0
        panel(axes[row, 0], pet2, mip(a), "autumn", f"M0  ({int(ndi.label(a)[1])} components)", vmax)
        panel(axes[row, 1], pet2, mip(b), "autumn",
              f"M1 = {vv}  ({int(ndi.label(b)[1])} components)", vmax)
        panel(axes[row, 2], pet2, mip(gt), "winter",
              "ground truth" + (" (none)" if not gt.any() else ""), vmax)
        axes[row, 0].set_ylabel(f"{case}\n{caption}", fontsize=8)
    fig.suptitle("Experiment 4: coronal MIP before and after refinement", fontsize=11)
    fig.tight_layout()
    out = args.out / "phase3_examples.png"
    fig.savefig(out, dpi=150)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
