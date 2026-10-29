#!/usr/bin/env python3
"""Piece 3 confirmatory evaluation: case-level Dice, FP-clearance, lesion
sensitivity for the class-balanced ranking policy, out-of-fold and on
validation. Same read/boot functions and recipe as piece 1/2's eval, new
reference row added for comparability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice, mcnemar_fp, lesion_sens

R = Path(r"D:\data\reports")

REFERENCE = {
    "(a) fixed topology_remove": {"delta": 0.0033, "lo": -0.0035, "hi": 0.0099,
                                   "fp_cleared": "230/385"},
    "(b) hand-built policy": {"delta": 0.0065, "lo": 0.0025, "hi": 0.0118,
                               "fp_cleared": "231/385"},
    "(c) learned action-value model": {"delta": 0.0052, "lo": 0.0022, "hi": 0.0099,
                                        "fp_cleared": "237/385"},
    "(d) shrinkage policy (piece 1, falsified)": {"delta": 0.0029, "lo": 0.0015,
                                                    "hi": 0.0048, "fp_cleared": "50/385"},
    "(e) pairwise ranking (piece 2)": {"delta": 0.0033, "lo": -0.0035, "hi": 0.0099,
                                        "fp_cleared": "59/385"},
}


def run(prefix: str, label: str) -> dict:
    base = read(R / prefix, "base")
    fin = read(R / prefix, "final")
    d = paired_dice(fin, base)
    fp = mcnemar_fp(fin, base)
    sens_base = lesion_sens(base)
    sens_fin = lesion_sens(fin)
    out = {
        "label": label,
        "n_pos": d["n"],
        "delta_dice": d["delta"], "lo": d["lo"], "hi": d["hi"], "p": d["p"],
        "dice_base": d["b"], "dice_agent": d["a"],
        "fp_cleared": fp["cleared"], "fp_added": fp["added"],
        "fp_n_neg": fp["n_neg"], "fp_p": fp["p"],
        "lesion_sens_base": sens_base, "lesion_sens_agent": sens_fin,
        "lesion_sens_delta": sens_fin - sens_base,
    }
    print(f"\n{label}")
    print(f"  n (tumour-positive) = {d['n']}")
    print(f"  dDice = {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p']:.4f}")
    print(f"  tumour-free FP cleared: {fp['cleared']}/{fp['n_neg']}  added: {fp['added']}")
    print(f"  lesion sensitivity: {sens_base:.4f} -> {sens_fin:.4f}  "
          f"(delta {sens_fin - sens_base:+.4f}, safety margin -0.01 binding)")
    return out


def main() -> None:
    results = {"oof": run("p7_oof_balanced", "Piece 3 (class-balanced ranking), out-of-fold"),
                "val": run("p7_val_balanced", "Piece 3 (class-balanced ranking), validation")}

    print("\n" + "=" * 78)
    print("Comparison against reference rows (cross-fitted, OOF, T=10, p_failure>=0.90)")
    print("=" * 78)
    for name, ref in REFERENCE.items():
        print(f"  {name:42s} dDice {ref['delta']:+.4f} [{ref['lo']:+.4f}, {ref['hi']:+.4f}]  "
              f"FP cleared {ref['fp_cleared']}")
    oof = results["oof"]
    print(f"  {'(f) class-balanced ranking (this run)':42s} dDice {oof['delta_dice']:+.4f} "
          f"[{oof['lo']:+.4f}, {oof['hi']:+.4f}]  "
          f"FP cleared {oof['fp_cleared']}/{oof['fp_n_neg']}")

    bar = REFERENCE["(b) hand-built policy"]
    piece2 = REFERENCE["(e) pairwise ranking (piece 2)"]
    beats_piece2 = oof["delta_dice"] > piece2["delta"] or \
        int(oof["fp_cleared"]) > int(piece2["fp_cleared"].split("/")[0])
    beats_bar = oof["delta_dice"] > bar["delta"] and oof["lo"] > 0
    verdict = (f"beats piece 2 (dDice or FP-cleared): {beats_piece2}; "
               f"beats hand-built policy (b), cross-fitted: {beats_bar}")
    print(f"\nVerdict: {verdict}")

    out = {"reference": REFERENCE, "results": results,
           "beats_piece2": beats_piece2, "beats_handbuilt": beats_bar,
           "verdict": verdict}
    out_path = R / "phase7_piece3_eval.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
