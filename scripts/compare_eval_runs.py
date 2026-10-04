#!/usr/bin/env python3
"""Put two evaluation runs side by side from the per-case CSVs.

Reads the CSVs `evaluate_predictions.py` writes, so a comparison can be redone
later without re-running inference.

Two counts are reported where the evaluator prints one, because they are not the
same thing: a prediction that is *empty* ("complete miss" in the evaluator's
summary), and a prediction that is non-empty but overlaps the lesion nowhere.
Dice 0 is the union of the two, and on a model that over-predicts, most Dice-0
cases are the second kind.

    python scripts/compare_eval_runs.py --a-label 503 --b-label 505 \\
        --pair test reports/eval_503_test.csv reports/eval_505_test.csv \\
        --pair val  reports/eval_503_val.csv  reports/eval_505_val.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROWS = [("Dice (tumour-positive)", "dice", "{:.4f}", "ratio"),
        ("  median Dice", "dice_med", "{:.4f}", "ratio"),
        ("  positives scoring Dice 0", "miss", "{:d}", "diff"),
        ("  of those, empty prediction", "empty", "{:d}", "diff"),
        ("HD95 mm (mean)", "hd", "{:.1f}", "pct"),
        ("  median HD95 mm", "hd_med", "{:.1f}", "pct"),
        ("FP volume mL (all cases)", "fp", "{:.2f}", "pct"),
        ("FN volume mL (all cases)", "fn", "{:.2f}", "pct")]


def stats(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pos = [r for r in rows if r["gt_positive"] == "1"]
    neg = [r for r in rows if r["gt_positive"] == "0"]
    dice = np.array([float(r["dice"]) for r in pos])
    hd = np.array([float(r["hd95_mm"]) for r in pos
                   if r["hd95_mm"] not in ("", "nan")
                   and np.isfinite(float(r["hd95_mm"]))])
    fp = np.array([float(r["fp_volume_ml"]) for r in rows])
    fn = np.array([float(r["fn_volume_ml"]) for r in rows])
    return {"n": len(rows), "pos": len(pos),
            "dice": float(dice.mean()), "dice_med": float(np.median(dice)),
            "empty": sum(1 for r in pos if r["pred_positive"] == "0"),
            "miss": int((dice == 0).sum()),
            "hd": float(hd.mean()), "hd_med": float(np.median(hd)),
            "fp": float(fp.mean()), "fn": float(fn.mean()),
            "negfp": sum(1 for r in neg if float(r["fp_volume_ml"]) > 0),
            "neg": len(neg)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-label", default="A")
    p.add_argument("--b-label", default="B")
    p.add_argument("--pair", nargs=3, action="append", required=True,
                   metavar=("NAME", "A_CSV", "B_CSV"))
    args = p.parse_args()

    for name, a_csv, b_csv in args.pair:
        sa, sb = stats(Path(a_csv)), stats(Path(b_csv))
        if (sa["n"], sa["pos"]) != (sb["n"], sb["pos"]):
            print(f"WARNING: {name} compares different case sets: "
                  f"{sa['n']}/{sa['pos']} vs {sb['n']}/{sb['pos']}")
        print(f"\n### {name}  ({sa['n']} cases, {sa['pos']} tumour-positive)")
        print(f"{'metric':28s} {args.a_label:>14s} {args.b_label:>14s} {'change':>12s}")
        print("-" * 72)
        for label, key, fmt, how in ROWS:
            va, vb = sa[key], sb[key]
            if how == "diff":
                chg = f"{vb - va:+d}"
            elif not va:
                chg = "n/a"
            elif how == "ratio":
                chg = f"x{vb / va:.2f}"
            else:
                chg = f"{(vb - va) / va * 100:+.0f}%"
            print(f"{label:28s} {fmt.format(va):>14s} {fmt.format(vb):>14s} {chg:>12s}")
        a_negfp = f"{sa['negfp']}/{sa['neg']}"
        b_negfp = f"{sb['negfp']}/{sb['neg']}"
        print(f"{'tumour-free with any FP':28s} {a_negfp:>14s} {b_negfp:>14s} "
              f"{sb['negfp'] - sa['negfp']:>+12d}")


if __name__ == "__main__":
    main()
