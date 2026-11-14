#!/usr/bin/env python3
"""Phase XXI, Part C: a safeguarded switch from the deployed output to the
fine-tuned segmenter.

See `reports/phase21_baselines_shift_safeguard_plan.md`. Post hoc: selection
on B, a check on C. Not a certificate.

    S_tau = {p_FT > 0.5}  u  (P19 n {p_FT >= tau}),   S_0.5 = FT,  S_0 = FT u P19

  --stage B   checks, the tau table on B, tau* per delta.
              Writes phase21_safeswitch_B.json. No C file is opened.
  --stage C   after reports/phase21_lock.md exists. The same table on C and
              the outcome of each tau*. Writes phase21_safeswitch_C.json.

Inputs (all existing or regenerated under new names):
  C:\\psma_maps\\ft                 fine-tuned ensemble maps
  D:\\data\\reports\\p21_masks_{B,C}  P19 masks from phase5_agent.py --save-masks
  p21{B,C}_psma_p19_final.csv      that run's table; must equal the locked one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import MIN_VOX, label_path
from phase5_eval import read, paired_dice

R = Path(r"D:\data\reports")
REPO = Path(__file__).resolve().parent.parent
FT_MAPS = Path(r"C:\psma_maps\ft")
ALPHA = 0.05
DELTAS = [0.05, 0.10, 0.15, 0.20]
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]   # safest first
LOCKED = {"B": {"P19": "p20B_psma_p19", "FT": ("p20B_ft_hb", "base"), "new": "p21B_psma_p19"},
          "C": {"P19": "p19_psma_p19", "FT": ("p20_ft_p20", "base"), "new": "p21C_psma_p19"}}


def ub(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))


def score(m, gt, g_lab, g_n, vml):
    """Dice, lesion hits and FP volume, as CaseContext.score."""
    pv, gv = int(m.sum()), int(gt.sum())
    hit = set(np.unique(g_lab[m]).tolist()) - {0}
    lab, n = ndi.label(m)
    fp = 0
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        touch = np.zeros(n + 1, bool)
        touch[np.unique(lab[gt])] = True
        fp = int(sizes[1:][(sizes[1:] >= MIN_VOX) & ~touch[1:]].sum())
    return {"dice": (2 * int((m & gt).sum()) / (pv + gv)) if (pv + gv) else float("nan"),
            "n_hit": len(hit), "n_gt": g_n, "fp_ml": fp * vml, "pred_ml": pv * vml}


def harmed(x, ref):
    return x["n_hit"] < ref["n_hit"] or (ref["n_gt"] > 0 and x["dice"] - ref["dice"] < -0.01)


def pos_dice(rows):
    return float(np.mean([r["dice"] for r in rows.values() if r["n_gt"] > 0]))


def same(a, b, keys=("dice", "n_hit", "fp_ml"), tol=1e-9):
    for k in keys:
        x, y = float(a[k]), float(b[k])
        if not ((x != x and y != y) or abs(x - y) < tol):
            return False
    return True


def run(split: str) -> dict:
    cases = json.loads((REPO / "reports" / "phase19_split.json").read_text())[split]
    L = LOCKED[split]
    p19 = read(R / L["P19"], "final")
    ft = read(R / L["FT"][0], L["FT"][1])
    new = read(R / L["new"], "final")
    assert sorted(p19) == sorted(ft) == sorted(new) == sorted(cases)
    bad = [c for c in cases if not same(new[c], p19[c])]
    assert not bad, f"regenerated P19 table differs from the locked one: {bad[:3]}"
    print(f"[{split}] check 1 passed: regenerated P19 table equals the locked one ({len(cases)} cases)")

    S = {t: {} for t in TAUS}
    for i, c in enumerate(cases, 1):
        z = np.load(FT_MAPS / f"{c}.npz")
        p = z["p_bar"].astype(np.float32)
        vml = float(np.prod(z["spacing"])) / 1000.0
        mz = np.load(R / f"p21_masks_{split}" / f"{c}.npz")
        dep = mz["mask"].astype(bool)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
        assert p.shape == dep.shape == gt.shape, c
        g_lab, g_n = ndi.label(gt)
        s19 = score(dep, gt, g_lab, g_n, vml)
        assert same(s19, p19[c], ("dice", "n_hit")), f"P19 mask does not reproduce its row: {c}"
        ftm = p > 0.5
        for t in TAUS:
            m = ftm if t == 0.5 else (ftm | (dep & (p >= t)))
            S[t][c] = score(m, gt, g_lab, g_n, vml)
        assert same(S[0.5][c], ft[c]), f"S_0.5 does not reproduce the locked FT row: {c}"
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)
    print(f"[{split}] checks 2 and 3 passed: P19 masks and S_0.5 reproduce the locked rows")

    n = len(cases)
    rep = {"split": split, "n": n, "P19_dice_pos": pos_dice(p19), "FT_dice_pos": pos_dice(ft),
           "P19_fp_ml": float(np.mean([p19[c]["fp_ml"] for c in cases])),
           "P19_hits": int(sum(p19[c]["n_hit"] for c in cases)), "taus": []}
    print(f"  {'tau':>5s} {'harmed':>7s} {'UB95':>6s} {'lost-hit pts':>12s} {'Dice':>7s} "
          f"{'FP mL':>8s} {'hits':>6s}")
    for t in TAUS:
        h = [c for c in cases if harmed(S[t][c], p19[c])]
        lost = [c for c in h if S[t][c]["n_hit"] < p19[c]["n_hit"]]
        row = {"tau": t, "k": len(h), "ub95": ub(len(h), n), "k_lost_hit": len(lost),
               "p": {str(d): float(binom.cdf(len(h), n, d)) for d in DELTAS},
               "dice_pos": pos_dice(S[t]),
               "fp_ml": float(np.mean([S[t][c]["fp_ml"] for c in cases])),
               "hits": int(sum(S[t][c]["n_hit"] for c in cases)),
               "vs_P19": paired_dice(S[t], p19), "vs_FT": paired_dice(S[t], ft),
               "harmed_cases": h}
        rep["taus"].append(row)
        print(f"  {t:5.2f} {len(h):4d}/{n} {row['ub95']:6.3f} {len(lost):12d} "
              f"{row['dice_pos']:7.4f} {row['fp_ml']:8.1f} {row['hits']:6d}")
    print(f"  P19   {'':7s} {'':6s} {'':12s} {rep['P19_dice_pos']:7.4f} {rep['P19_fp_ml']:8.1f} "
          f"{rep['P19_hits']:6d}")
    return rep


def select(rep: dict) -> list:
    """Fixed sequence from tau = 0 upward at alpha; best mean Dice among those that pass."""
    out = []
    for d in DELTAS:
        ok = []
        for row in rep["taus"]:
            if row["p"][str(d)] >= ALPHA:
                break
            ok.append(row)
        best = max(ok, key=lambda r: r["dice_pos"]) if ok else None
        if best is not None and not best["dice_pos"] > rep["P19_dice_pos"]:
            best = None
        out.append({"delta": d, "passed": [r["tau"] for r in ok],
                    "tau_star": best["tau"] if best else None})
        print(f"  delta {d:.2f}: passed {[r['tau'] for r in ok]} -> "
              f"tau* = {best['tau'] if best else 'rollback'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("B", "C"), required=True)
    st = ap.parse_args().stage
    if st == "B":
        rep = run("B")
        rep["selection"] = select(rep)
        (R / "phase21_safeswitch_B.json").write_text(json.dumps(rep, indent=1, default=float))
        return
    assert (REPO / "reports" / "phase21_lock.md").exists(), "commit the lock before reading C"
    sel = json.loads((R / "phase21_safeswitch_B.json").read_text())["selection"]
    rep = run("C")
    by = {r["tau"]: r for r in rep["taus"]}
    rep["selected"] = []
    for s in sel:
        r = by.get(s["tau_star"])
        rep["selected"].append({"delta": s["delta"], "tau_star": s["tau_star"],
                                **({k: r[k] for k in ("k", "ub95", "k_lost_hit", "dice_pos",
                                                      "fp_ml", "hits", "vs_P19", "vs_FT")}
                                   if r else {"k": 0, "rollback": True})})
        if r:
            print(f"  delta {s['delta']:.2f}: tau* {s['tau_star']} -> harmed {r['k']}/{rep['n']} "
                  f"(UB {r['ub95']:.3f}), Dice {r['dice_pos']:.4f}, "
                  f"vs FT {r['vs_FT']['delta']:+.4f}")
        else:
            print(f"  delta {s['delta']:.2f}: rollback")
    (R / "phase21_safeswitch_C.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
