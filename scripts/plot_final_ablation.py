#!/usr/bin/env python3
"""Regenerate `docs/results/fig_final_ablation.png` (work plan section 21).

The figure was originally produced by an inline command that was never
committed, so this script exists to make it reproducible. It draws exactly what
that command drew, from the same `final_eval_tables.json`, with one change: the
Dice axis now reaches 0.71 so the validation bars (0.7061-0.7068) are not
clipped by the old 0.700 limit. No number is recomputed here - every value is
read from the JSON the analysis wrote.

    python scripts/plot_final_ablation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["M0", "unc", "uncmatch", "reason", "refine", "reflect", "iterate",
         "learned", "finalagent"]
LABELS = ["baseline", "+unc\n(0.90)", "+unc\n(matched)", "+reasoning",
          "+refinement", "+reflection\n(Q gate)", "+iteration\n(T=10)",
          "+learned\nmemory", "FINAL\nagent"]
SPLITS = (("oof", "#2c3e50"), ("val", "#7f8c8d"), ("test", "#c0392b"))

# Raised from 0.700: the validation bars sit at 0.7061-0.7068 and were clipped.
DICE_YLIM = (0.655, 0.710)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path,
                    default=Path(r"D:\data\reports\final_eval_tables.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "docs" / "results" / "fig_final_ablation.png")
    args = ap.parse_args()

    T = json.load(open(args.tables, encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(ORDER))
    w = 0.27

    for k, (sp, c) in enumerate(SPLITS):
        v = [T["section21"][sp][t]["dice"] for t in ORDER]
        ax[0].bar(x + (k - 1) * w, v, w, color=c,
                  label=f"{sp}{' (descriptive)' if sp == 'test' else ''}")
        n = [T["section21"][sp][t]["neg_with_fp"] for t in ORDER]
        tot = T["section21"][sp]["M0"]["n_neg"]
        ax[1].bar(x + (k - 1) * w, [100 * a / tot for a in n], w, color=c)

    ax[0].set_ylim(*DICE_YLIM)
    ax[0].set_ylabel("mean Dice, tumour-positive")
    ax[0].set_title("Section 21 ablation: Dice")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_ylabel("% of tumour-free patients with any FP")
    ax[1].set_title("Section 21 ablation: false-positive burden")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(LABELS, fontsize=7.5)
        a.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    lo = min(T["section21"][sp][t]["dice"] for sp, _ in SPLITS for t in ORDER)
    hi = max(T["section21"][sp][t]["dice"] for sp, _ in SPLITS for t in ORDER)
    print(f"written {args.out}")
    print(f"Dice range plotted {lo:.4f} to {hi:.4f}, axis {DICE_YLIM[0]} to "
          f"{DICE_YLIM[1]} -> {'no clipping' if hi <= DICE_YLIM[1] else 'STILL CLIPPED'}")


if __name__ == "__main__":
    main()
