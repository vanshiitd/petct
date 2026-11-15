#!/usr/bin/env python3
"""Phase XXII: the fine-tuned segmenter at lower operating points,
FT_t = {p_FT > t}, against the deployed system P19.

See `reports/phase22_operating_point_plan.md`.

  --stage B   reproduction check, the threshold table on B, t* per delta and
              alpha_B. Writes phase22_threshold_B.json. No C file is opened.
  --stage C   only after reports/phase22_lock.md exists. T1-T3 for t* and the
              descriptive table. Writes phase22_threshold_C.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import label_path
from phase5_eval import read, paired_dice
from phase21_safeswitch import LOCKED, R, REPO, FT_MAPS, score, harmed, pos_dice, same, ub

TS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]      # most sensitive first
DELTAS = [0.05, 0.10, 0.15, 0.20]
ALPHAS = [0.05, 0.025]


def run(split: str) -> tuple[dict, dict]:
    cases = json.loads((REPO / "reports" / "phase19_split.json").read_text())[split]
    L = LOCKED[split]
    p19 = read(R / L["P19"], "final")
    ft = read(R / L["FT"][0], L["FT"][1])
    S = {t: {} for t in TS}
    for i, c in enumerate(cases, 1):
        z = np.load(FT_MAPS / f"{c}.npz")
        p = z["p_bar"].astype(np.float32)
        vml = float(np.prod(z["spacing"])) / 1000.0
        dep = np.load(R / f"p21_masks_{split}" / f"{c}.npz")["mask"].astype(bool)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
        g_lab, g_n = ndi.label(gt)
        assert same(score(dep, gt, g_lab, g_n, vml), p19[c], ("dice", "n_hit")), c
        for t in TS:
            S[t][c] = score(p > t, gt, g_lab, g_n, vml)
        assert same(S[0.5][c], ft[c]), f"FT_0.5 does not reproduce the locked FT row: {c}"
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)
    print(f"[{split}] checks passed: P19 masks and FT_0.5 reproduce the locked rows")
    n = len(cases)
    rep = {"split": split, "n": n, "P19_dice_pos": pos_dice(p19),
           "P19_fp_ml": float(np.mean([p19[c]["fp_ml"] for c in cases])),
           "P19_hits": int(sum(p19[c]["n_hit"] for c in cases)), "ts": []}
    print(f"  {'t':>5s} {'harmed':>7s} {'UB95':>6s} {'lost-hit':>8s} {'Dice':>7s} {'FP mL':>7s} "
          f"{'hits':>6s} {'dDice vs FT_0.5 [95%]':>28s}")
    for t in TS:
        h = [c for c in cases if harmed(S[t][c], p19[c])]
        vs = paired_dice(S[t], S[0.5])
        row = {"t": t, "k": len(h), "ub95": ub(len(h), n),
               "k_lost_hit": sum(S[t][c]["n_hit"] < p19[c]["n_hit"] for c in h),
               "p": {str(d): float(binom.cdf(len(h), n, d)) for d in DELTAS},
               "dice_pos": pos_dice(S[t]),
               "fp_ml": float(np.mean([S[t][c]["fp_ml"] for c in cases])),
               "hits": int(sum(S[t][c]["n_hit"] for c in cases)),
               "pred_ml": float(np.mean([S[t][c]["pred_ml"] for c in cases])),
               "vs_FT05": vs, "vs_P19": paired_dice(S[t], p19), "harmed_cases": h}
        rep["ts"].append(row)
        print(f"  {t:5.2f} {len(h):4d}/{n} {row['ub95']:6.3f} {row['k_lost_hit']:8d} "
              f"{row['dice_pos']:7.4f} {row['fp_ml']:7.1f} {row['hits']:6d} "
              f"{vs['delta']:+.4f} [{vs['lo']:+.4f}, {vs['hi']:+.4f}]")
    print(f"  P19   {'':7s} {'':6s} {'':8s} {rep['P19_dice_pos']:7.4f} {rep['P19_fp_ml']:7.1f} "
          f"{rep['P19_hits']:6d}")
    return rep, S


def select(rep: dict) -> list:
    out = []
    for a in ALPHAS:
        for d in DELTAS:
            ok = []
            for row in rep["ts"]:
                if row["p"][str(d)] >= a:
                    break
                ok.append(row)
            best = max(ok, key=lambda r: r["dice_pos"]) if ok else None
            out.append({"alpha_B": a, "delta": d, "passed": [r["t"] for r in ok],
                        "t_star": best["t"] if best else None})
            print(f"  alpha_B {a} delta {d:.2f}: passed {[r['t'] for r in ok]} -> "
                  f"t* = {best['t'] if best else 'rollback'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("B", "C"), required=True)
    st = ap.parse_args().stage
    if st == "B":
        rep, _ = run("B")
        rep["selection"] = select(rep)
        (R / "phase22_threshold_B.json").write_text(json.dumps(rep, indent=1, default=float))
        return
    assert (REPO / "reports" / "phase22_lock.md").exists(), "commit the lock before reading C"
    sel = json.loads((R / "phase22_threshold_B.json").read_text())["selection"]
    rep, _ = run("C")
    by = {r["t"]: r for r in rep["ts"]}
    rep["tests"] = []
    for s in sel:
        r = by.get(s["t_star"])
        if r is None:
            rep["tests"].append({**s, "rollback": True})
            print(f"  alpha_B {s['alpha_B']} delta {s['delta']:.2f}: rollback")
            continue
        p = float(binom.cdf(r["k"], rep["n"], s["delta"]))
        t1, t2 = p < 0.05, r["vs_FT05"]["lo"] > -0.01
        rep["tests"].append({**s, "k": r["k"], "p": p, "ub95": r["ub95"], "T1": bool(t1),
                             "T2": bool(t2), "vs_FT05": r["vs_FT05"], "vs_P19": r["vs_P19"],
                             "dice_pos": r["dice_pos"], "fp_ml": r["fp_ml"], "hits": r["hits"]})
        print(f"  alpha_B {s['alpha_B']} delta {s['delta']:.2f}: t* {s['t_star']} -> harmed "
              f"{r['k']}/{rep['n']} p={p:.4f} UB {r['ub95']:.3f} T1 {'met' if t1 else 'FAILED'} | "
              f"dDice vs FT_0.5 {r['vs_FT05']['delta']:+.4f} [{r['vs_FT05']['lo']:+.4f}, "
              f"{r['vs_FT05']['hi']:+.4f}] T2 {'met' if t2 else 'FAILED'} | vs P19 "
              f"{r['vs_P19']['delta']:+.4f}")
    (R / "phase22_threshold_C.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
