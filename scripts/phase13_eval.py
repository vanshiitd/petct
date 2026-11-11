#!/usr/bin/env python3
"""Phase XIII evaluation, exactly as `reports/phase13_context_detector_plan.md`
fixed it, paired case by case against the hand-built run:

  1 FP-clearance superiority  more tumour-free cleared, exact McNemar p < 0.025
  2 Dice superiority          paired dDice > 0, bootstrap p < 0.025
  3 Dice non-inferiority      paired dDice lower 95% >= -0.0025
  4 lesion non-inferiority    lesion hits retained, case bootstrap lower 95% >= -5
  5 beats hand-built overall  (1 or 2) and 3 and 4

Validation direction check: cleared >= hand-built, dDice >= -0.0025, lesion
hits retained >= -1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice
from phase11_eval import boot_sum, summary, tertiles, subgroup_rows

R = Path(r"D:\data\reports")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oof")
    ap.add_argument("--run", default=None)
    ap.add_argument("--hb", default=None)
    ap.add_argument("--tag", default="phase13")
    args = ap.parse_args()
    sp = args.split
    run_p = args.run or f"p13_{sp}_det"
    hb_p = args.hb or f"au_{sp}_attr_policy"

    base = read(R / hb_p, "base")
    hb = read(R / hb_p, "final")
    a = read(R / run_p, "final")
    assert set(a) == set(hb), "runs cover different cases"
    pos = sorted(c for c in base if base[c]["n_gt"] > 0)
    neg = sorted(c for c in base if base[c]["n_gt"] == 0)

    ca = {c: a[c]["n_fp_comp"] == 0 and base[c]["n_fp_comp"] > 0 for c in neg}
    ch = {c: hb[c]["n_fp_comp"] == 0 and base[c]["n_fp_comp"] > 0 for c in neg}
    only_a = sum(ca[c] and not ch[c] for c in neg)
    only_h = sum(ch[c] and not ca[c] for c in neg)
    p_mc = binomtest(only_a, only_a + only_h, 0.5).pvalue if only_a + only_h else 1.0
    d_hit = np.array([a[c]["n_hit"] - hb[c]["n_hit"] for c in pos], float)
    s, lo, hi = boot_sum(d_hit)
    head = paired_dice(a, hb)
    S_a, S_h = summary(a, base), summary(hb, base)

    if sp == "val":
        crit = {"cleared_ge_hb": S_a["fp_cleared"] >= S_h["fp_cleared"],
                "dice_ge_margin": head["delta"] >= -0.0025,
                "lesions_ge_-1": s >= -1}
        crit["direction_check"] = all(crit.values())
    else:
        crit = {"1_fp_superiority": only_a > only_h and p_mc < 0.025,
                "2_dice_superiority": head["delta"] > 0 and head["p"] < 0.025,
                "3_dice_ni": head["lo"] >= -0.0025,
                "4_lesion_ni": lo >= -5}
        crit["5_beats_handbuilt"] = bool((crit["1_fp_superiority"] or crit["2_dice_superiority"])
                                         and crit["3_dice_ni"] and crit["4_lesion_ni"])

    print(f"=== {args.tag}, {sp}: {run_p} vs {hb_p}, paired "
          f"({len(pos)} positive / {len(neg)} tumour-free) ===")
    print(f"  tumour-free cleared: {S_a['fp_cleared']} vs HB {S_h['fp_cleared']}  "
          f"discordant {only_a} vs {only_h}  McNemar p={p_mc:.4g}  "
          f"(new FP: {S_a['fp_added']} vs {S_h['fp_added']})")
    print(f"  dDice vs HB: {head['delta']:+.4f} [{head['lo']:+.4f}, {head['hi']:+.4f}]  "
          f"p={head['p']:.4g}")
    print(f"  lesion hits retained vs HB: {s:+.0f} [{lo:+.0f}, {hi:+.0f}]")
    for name, S in (("this run", S_a), ("hand-built", S_h)):
        print(f"  {name:11s} vs M0: dDice {S['dDice_vs_M0']:+.4f} [{S['lo']:+.4f}, "
              f"{S['hi']:+.4f}]  lesions lost {S['lesions_lost']:.0f}  "
              f"cleared {S['fp_cleared']}/{S['n_neg']}")
    print(f"  criteria: {crit}")

    ob = read(R / "au_oof_attr_policy", "base")
    opos = [c for c in ob if ob[c]["n_gt"] > 0]
    bounds = {k: tertiles([ob[c][k] for c in opos]) for k in ("n_gt", "dice")}
    sub = subgroup_rows(a, hb, base, bounds)
    flags = [g for g, r in sub.items() if r["flag"]]
    print(f"  subgroups flagged (harm vs M0): {flags or 'none'}")

    out = R / f"{args.tag}_eval_{sp}.json"
    out.write_text(json.dumps({
        "split": sp, "run": run_p, "handbuilt": hb_p,
        "fp_discordant": {"only_run": only_a, "only_hb": only_h, "p": p_mc},
        "dDice_vs_hb": head, "lesions_retained_vs_hb": {"sum": s, "lo": lo, "hi": hi},
        "run_vs_M0": S_a, "hb_vs_M0": S_h, "criteria": crit, "subgroups": sub},
        indent=1, default=float), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
