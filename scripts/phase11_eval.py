#!/usr/bin/env python3
"""Phase XI evaluation, exactly as `reports/phase11_experience_gate_plan.md`
fixed it: paired, case by case, against the hand-built run.

  1 lesion-safety superiority   sum over tumour-positive cases of lesion hits
                                (EAG - hand-built), case bootstrap, lower 95% > 0
  2 Dice non-inferiority        paired dDice (EAG - hand-built), lower 95% >= -0.0025
  3 FP-clearance                tumour-free cleared >= hand-built's - 2
  4 win                         1-3 and dDice point estimate >= 0
  5 secondary                   against the final agent (Q gate)

plus the pre-specified proxy-subgroup no-harm table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice, mcnemar_fp, lesion_sens

R = Path(r"D:\data\reports")
NBOOT = 2000
DICE_NI = -0.0025
FP_SLACK = 2


def boot_sum(d, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].sum() for _ in range(NBOOT)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.sum()), float(lo), float(hi)


def lesions_lost(fin, base):
    return float(sum(base[c]["n_hit"] - fin[c]["n_hit"] for c in base
                     if base[c]["n_gt"] > 0 and c in fin))


def summary(fin, base):
    d = paired_dice(fin, base)
    fp = mcnemar_fp(fin, base)
    return {"dDice_vs_M0": d["delta"], "lo": d["lo"], "hi": d["hi"],
            "fp_cleared": fp["cleared"], "fp_added": fp["added"], "n_neg": fp["n_neg"],
            "lesions_lost": lesions_lost(fin, base),
            "lesion_sens": lesion_sens(fin), "lesion_sens_M0": lesion_sens(base)}


def tertiles(vals):
    return [float(x) for x in np.quantile(vals, [1 / 3, 2 / 3])]


def subgroup_rows(eag, hb, base, bounds):
    pos = [c for c in base if base[c]["n_gt"] > 0]
    groups = {"tumour-free": [c for c in base if base[c]["n_gt"] == 0],
              "tumour-positive": pos}
    for name, key in (("lesion count", "n_gt"), ("M0 Dice", "dice")):
        lo, hi = bounds[key]
        v = {c: base[c][key] for c in pos}
        groups[f"{name} low"] = [c for c in pos if v[c] <= lo]
        groups[f"{name} mid"] = [c for c in pos if lo < v[c] <= hi]
        groups[f"{name} high"] = [c for c in pos if v[c] > hi]
    out = {}
    for g, cs in groups.items():
        sub = lambda d: {c: d[c] for c in cs if c in d}
        b, e, h = sub(base), sub(eag), sub(hb)
        row = {"n": len(cs)}
        if g == "tumour-free":
            fp_e, fp_h = mcnemar_fp(e, b), mcnemar_fp(h, b)
            row.update({"cleared_eag": fp_e["cleared"], "added_eag": fp_e["added"],
                        "cleared_hb": fp_h["cleared"], "added_hb": fp_h["added"],
                        "flag": fp_e["added"] > fp_e["cleared"]})
        else:
            dm, dh = paired_dice(e, b), paired_dice(e, h)
            row.update({"dDice_vs_M0": dm["delta"], "lo_M0": dm["lo"], "hi_M0": dm["hi"],
                        "dDice_vs_hb": dh["delta"], "lo_hb": dh["lo"], "hi_hb": dh["hi"],
                        "lost_eag": lesions_lost(e, b), "lost_hb": lesions_lost(h, b),
                        "flag": dm["hi"] < 0})
        out[g] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oof")
    ap.add_argument("--eag", default=None)
    ap.add_argument("--hb", default=None)
    ap.add_argument("--final", default=None)
    args = ap.parse_args()
    sp = args.split
    eag_p = args.eag or f"p11_{sp}_eag"
    hb_p = args.hb or f"au_{sp}_attr_policy"
    fa_p = args.final or f"au_{sp}_finalagent"

    base = read(R / hb_p, "base")
    hb = read(R / hb_p, "final")
    eag = read(R / eag_p, "final")
    assert set(eag) == set(hb), "runs cover different cases"
    pos = sorted(c for c in base if base[c]["n_gt"] > 0)

    d_hit = np.array([eag[c]["n_hit"] - hb[c]["n_hit"] for c in pos], float)
    s, lo, hi = boot_sum(d_hit)
    head = paired_dice(eag, hb)
    S_e, S_h = summary(eag, base), summary(hb, base)

    c1 = lo > 0
    c2 = head["lo"] >= DICE_NI
    c3 = S_e["fp_cleared"] >= S_h["fp_cleared"] - FP_SLACK
    win = c1 and c2 and c3 and head["delta"] >= 0
    res = {"split": sp, "runs": {"eag": eag_p, "handbuilt": hb_p, "final": fa_p},
           "lesions_retained_vs_hb": {"sum": s, "lo": lo, "hi": hi},
           "dDice_vs_hb": head, "eag": S_e, "handbuilt": S_h,
           "criteria": {"1_safety": c1, "2_dice_ni": c2, "3_fp_ni": c3, "4_win": win}}

    print(f"=== Phase XI, {sp}: EAG vs hand-built, paired ({len(pos)} tumour-positive) ===")
    print(f"  1 lesion hits retained (EAG - HB): {s:+.0f} [{lo:+.0f}, {hi:+.0f}]  -> {c1}")
    print(f"  2 dDice (EAG - HB): {head['delta']:+.4f} [{head['lo']:+.4f}, {head['hi']:+.4f}]"
          f"  p={head['p']:.4f}  -> {c2}")
    print(f"  3 tumour-free cleared: EAG {S_e['fp_cleared']}/{S_e['n_neg']} "
          f"(added {S_e['fp_added']})  HB {S_h['fp_cleared']}/{S_h['n_neg']} "
          f"(added {S_h['fp_added']})  -> {c3}")
    print(f"  4 WIN: {win}")
    for name, S in (("EAG", S_e), ("hand-built", S_h)):
        print(f"  {name:11s} vs M0: dDice {S['dDice_vs_M0']:+.4f} [{S['lo']:+.4f}, "
              f"{S['hi']:+.4f}]  lesions lost {S['lesions_lost']:.0f}  "
              f"sens {S['lesion_sens_M0']:.4f} -> {S['lesion_sens']:.4f}")

    fa_file = R / f"{fa_p}_final.csv"
    if fa_file.exists():
        fa = read(R / fa_p, "final")
        S_f = summary(fa, base)
        dom = S_e["dDice_vs_M0"] > S_f["dDice_vs_M0"] and S_e["lesions_lost"] <= S_f["lesions_lost"]
        res["final_agent"] = S_f
        res["criteria"]["5_beats_final_agent"] = dom
        head_f = paired_dice(eag, fa)
        res["dDice_vs_final"] = head_f
        print(f"  final agent vs M0: dDice {S_f['dDice_vs_M0']:+.4f} [{S_f['lo']:+.4f}, "
              f"{S_f['hi']:+.4f}]  lesions lost {S_f['lesions_lost']:.0f}  "
              f"cleared {S_f['fp_cleared']}/{S_f['n_neg']}")
        print(f"  5 EAG higher dDice and no more lesions lost than final agent: {dom}"
              f"   (paired EAG - final {head_f['delta']:+.4f} [{head_f['lo']:+.4f}, "
              f"{head_f['hi']:+.4f}])")

    # subgroup boundaries always from out-of-fold M0
    ob = read(R / "au_oof_attr_policy", "base")
    opos = [c for c in ob if ob[c]["n_gt"] > 0]
    bounds = {k: tertiles([ob[c][k] for c in opos]) for k in ("n_gt", "dice")}
    res["subgroup_bounds"] = bounds
    res["subgroups"] = subgroup_rows(eag, hb, base, bounds)
    print("  subgroups (proxy; flag = harm vs M0):")
    for g, r in res["subgroups"].items():
        if g == "tumour-free":
            print(f"    {g:20s} n={r['n']:3d}  cleared EAG {r['cleared_eag']} "
                  f"(+{r['added_eag']} new)  HB {r['cleared_hb']} (+{r['added_hb']})  "
                  f"flag {r['flag']}")
        else:
            print(f"    {g:20s} n={r['n']:3d}  vs M0 {r['dDice_vs_M0']:+.4f} "
                  f"[{r['lo_M0']:+.4f},{r['hi_M0']:+.4f}]  vs HB {r['dDice_vs_hb']:+.4f}  "
                  f"lost EAG {r['lost_eag']:.0f} HB {r['lost_hb']:.0f}  flag {r['flag']}")

    out = R / f"phase11_eval_{sp}.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
