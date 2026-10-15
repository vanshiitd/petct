#!/usr/bin/env python3
"""The three figures for the Phase II report.

  1. reliability diagram for P_bar, raw and temperature-scaled, in both regions
  2. case-level uncertainty against error, the scatter behind the Spearman values
  3. three example cases: PET with prediction, truth and uncertainty overlaid,
     one of them tumour-free with false positives

Slices are chosen by content, not fixed: the axial slice carrying the most
ground truth for a tumour-positive case, and the most predicted volume for a
tumour-free one, so the figure shows the thing it is supposed to illustrate.

    python scripts/exp2_figures.py --out-dir D:\\data\\reports
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import SimpleITK as sitk                  # noqa: E402

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
UNC = Path(r"D:\data\uncertainty")


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(case)


def fig_reliability(cal: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, region, title in zip(axes, ("near", "body"),
                                 ("within 10 mm of tumour (12% base rate)",
                                  "whole body (0.09% base rate)")):
        c = cal["test"]["calibration"][region]
        T = c["temperature"]
        for key, lab, style in (("bins_raw", "P_bar as predicted", "o-"),
                                ("bins_cal", f"after temperature T={T:.2f}", "s--")):
            b = [x for x in c[key] if x["n"] > 0 and np.isfinite(x["conf"])]
            ax.plot([x["conf"] for x in b], [x["freq"] for x in b], style, ms=4, lw=1.5,
                    label=f"{lab}  (ECE {c['ece_raw' if key == 'bins_raw' else 'ece_cal']:.4f})")
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls=":", label="perfect calibration")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed foreground frequency")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Calibration of the 2-model x 8-flip ensemble probability (test set)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def fig_scatter(cases: list[dict], ev: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    pos = [c for c in cases if c["gt_positive"] == 1 and c["case"] in ev]
    allc = [c for c in cases if c["case"] in ev]

    def panel(ax, xs, ys, xl, yl, title, logy=False, logx=False):
        ax.scatter(xs, ys, s=16, alpha=0.6, edgecolor="none", color="#1f77b4")
        from scipy import stats as st
        ok = np.isfinite(xs) & np.isfinite(ys)
        rho = st.spearmanr(xs[ok], ys[ok]).statistic
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{title}\nSpearman rho = {rho:.3f}  (n={int(ok.sum())})", fontsize=9)
        if logy:
            ax.set_yscale("symlog", linthresh=1)
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)

    x = np.array([c["u_stab_mean_pred"] for c in pos])
    y = np.array([1.0 - float(ev[c["case"]]["dice"]) for c in pos])
    panel(axes[0], x, y, "mean U_stab inside the prediction", "1 - Dice",
          "tumour-positive cases")

    x2 = np.array([c["u_stab_sum_pred"] for c in allc])
    y2 = np.array([float(ev[c["case"]]["fp_volume_ml"]) for c in allc])
    panel(axes[1], x2, y2, "summed U_stab inside the prediction (mL-weighted)",
          "false-positive volume (mL)", "all cases", logy=True, logx=True)

    x3 = np.array([c["u_stab_mean_pred"] for c in pos])
    y3 = np.array([float(ev[c["case"]]["hd95_mm"]) for c in pos])
    panel(axes[2], x3, y3, "mean U_stab inside the prediction", "HD95 (mm)",
          "tumour-positive cases")
    fig.suptitle("Case-level uncertainty against error, test set", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def fig_cases(picks: list[tuple[str, str, str]], out: Path) -> None:
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4.0 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for row, (split, case, caption) in enumerate(picks):
        z = np.load(UNC / split / f"{case}.npz")
        p_bar = z["p_bar"].astype(np.float32)
        u_stab = z["u_stab"].astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
        pred = p_bar > 0.5
        meta = json.loads((INV / f"{case}.json").read_text())
        suv = sitk.GetArrayFromImage(sitk.ReadImage(meta["source_pet"]))
        (z0, z1), (y0, y1), (x0, x1) = meta["crop_bbox_zyx"]
        suv = suv[z0:z1, y0:y1, x0:x1].astype(np.float32)

        # Coronal maximum-intensity projection, the conventional whole-body PET
        # view. A single coronal slice was tried first and is a poor choice: for
        # a case whose findings sit anteriorly the slice grazes the body edge and
        # shows mostly air, and any lesion off that plane disappears. Projecting
        # keeps every lesion visible and makes the three panels comparable.
        pet2 = suv.max(axis=1)
        gt2, pr2 = gt.any(axis=1), pred.any(axis=1)
        u2 = u_stab.max(axis=1)

        vmax = float(np.percentile(suv, 99.5)) or 1.0
        for col, (title, overlay, cmap, label) in enumerate([
                ("PET (SUV) + prediction", pr2, "autumn", "prediction"),
                ("PET (SUV) + ground truth", gt2, "winter", "ground truth"),
                ("U_stab (flip instability)", None, None, None)]):
            ax = axes[row, col]
            if overlay is not None:
                ax.imshow(pet2, cmap="gray_r", vmin=0, vmax=vmax, origin="lower",
                          aspect="auto")
                m = np.ma.masked_where(~overlay, overlay.astype(float))
                ax.imshow(m, cmap=cmap, alpha=0.65, origin="lower", aspect="auto",
                          vmin=0, vmax=1)
            else:
                im = ax.imshow(u2, cmap="magma", origin="lower", aspect="auto",
                               vmin=0, vmax=max(float(u2.max()), 1e-6))
                plt.colorbar(im, ax=ax, fraction=0.035)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(caption, fontsize=8)
    fig.suptitle("Coronal maximum-intensity projection: prediction, truth and "
                 "flip-instability uncertainty",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\data\reports"))
    args = ap.parse_args()
    R = args.out_dir
    print("figures:")

    cal = json.loads((R / "exp2_calibration.json").read_text())
    fig_reliability(cal, R / "phase2_reliability.png")

    cases = json.loads((R / "exp2_test_cases.json").read_text())
    with open(R / "eval505ens_test_full.csv", newline="", encoding="utf-8") as f:
        ev = {r["case"]: r for r in csv.DictReader(f)}
    fig_scatter(cases, ev, R / "phase2_uncertainty_vs_error.png")

    # pick three illustrative test cases from the evaluation itself
    rows = [ev[c["case"]] for c in cases]
    posr = [r for r in rows if r["gt_positive"] == "1" and np.isfinite(float(r["dice"]))]
    good = max(posr, key=lambda r: float(r["dice"]))
    partial = min((r for r in posr if float(r["dice"]) > 0.05),
                  key=lambda r: abs(float(r["dice"]) - 0.45))
    negr = [r for r in rows if r["gt_positive"] == "0"]
    worst_neg = max(negr, key=lambda r: float(r["fp_volume_ml"]))
    picks = [("test", good["case"], f"{good['case']}\nDice {float(good['dice']):.3f}"),
             ("test", partial["case"],
              f"{partial['case']}\nDice {float(partial['dice']):.3f}"),
             ("test", worst_neg["case"],
              f"{worst_neg['case']}\ntumour-free, FP "
              f"{float(worst_neg['fp_volume_ml']):.1f} mL")]
    for s, c, cap in picks:
        print(f"    example: {c} ({cap.splitlines()[1]})")
    fig_cases(picks, R / "phase2_example_cases.png")


if __name__ == "__main__":
    main()
