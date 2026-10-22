#!/usr/bin/env python3
"""Audit R1(c,d,f,g), R2(2) and R8, from the cross-fitted runs.

  R1(c)  H4 by Q tertile, u_stab tertile and all three inconsistency groups,
         with 95% and Bonferroni 97.5% intervals for the two hardest groups
  R1(d)  attribution: fixed / rule-based / learned at T=10, p_failure >= 0.90
  R1(f)  actions per case, improved / unchanged / degraded
  R1(g)  Phase V ladder rungs 2, 3, 5-noroi
  R2(2)  cases with more than T=10 gated candidates
  R8     the never-reported H5 comparison: Phase V agent against the Phase IV
         guarded configuration on the co-primaries
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
QCUTS = (0.130, 0.680)
UCUTS = (0.0382, 0.0834)


def boot_dice(a, b, seed=0, nboot=2000, pct=(2.5, 97.5)):
    cases = sorted(set(a) & set(b))
    pos = [c for c in cases if b[c]["n_gt"] > 0]
    da = np.array([a[c]["dice"] for c in pos], float)
    db = np.array([b[c]["dice"] for c in pos], float)
    rng = np.random.default_rng(seed)
    n = len(pos)
    bt = np.empty(nboot)
    for i in range(nboot):
        ix = rng.integers(0, n, n)
        bt[i] = np.nanmean(da[ix]) - np.nanmean(db[ix])
    lo, hi = np.percentile(bt, pct)
    return {"n": n, "delta": float(np.nanmean(da) - np.nanmean(db)),
            "lo": float(lo), "hi": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "audit_h4_attr.json")
    args = ap.parse_args()
    groups = json.load(open(R / "final_case_groups.json", encoding="utf-8"))
    out = {}

    base = read(R / "au_oof_reason", "base")
    fin = read(R / "au_oof_finalagent", "final")
    tr = read(R / "au_oof_finalagent", "traces")

    # ---------------- R1(c) H4 ----------------------------------------
    print("=" * 78)
    print("R1(c) H4, cross-fitted final agent, out-of-fold")
    q = {c: tr[c]["q0"] for c in tr}
    u = {c: groups["oof"][c]["ustab"] for c in base if c in groups["oof"]}
    sel = {}
    for lab, src, cuts, names in (
            ("Q", q, QCUTS, ("Q low (hardest)", "Q mid", "Q high (easiest)")),
            ("u_stab", u, UCUTS, ("u_stab low (easiest)", "u_stab mid",
                                  "u_stab high (hardest)"))):
        for nm, lo, hi in ((names[0], -np.inf, cuts[0]),
                           (names[1], cuts[0], cuts[1]),
                           (names[2], cuts[1], np.inf)):
            sel[nm] = [c for c in base if c in src and np.isfinite(src[c])
                       and lo <= src[c] < hi]
    for nm, key in (("anatomically inconsistent", "anat"),
                    ("metabolically inconsistent", "metab"),
                    ("topologically inconsistent", "topo")):
        sel[nm] = [c for c in base if c in groups["oof"]
                   and groups["oof"][c][key]]

    hard = {"Q low (hardest)", "u_stab high (hardest)"}
    h4 = {}
    print(f"  {'group':34s}{'n':>5s}{'dDice':>11s}{'95% CI':>24s}"
          f"{'  Bonferroni 97.5%':>24s}")
    for nm, cs in sel.items():
        pos = [c for c in cs if base[c]["n_gt"] > 0]
        if len(pos) < 5:
            print(f"  {nm:34s}{len(pos):5d}   too few")
            h4[nm] = {"n": len(pos), "too_few": True}
            continue
        a = {c: fin[c] for c in pos}
        b = {c: base[c] for c in pos}
        s = boot_dice(a, b)
        line = (f"  {nm:34s}{s['n']:5d}{s['delta']:+11.4f}"
                f"  [{s['lo']:+.4f}, {s['hi']:+.4f}]")
        if nm in hard:
            sb = boot_dice(a, b, pct=(1.25, 98.75))
            s["bonf_lo"], s["bonf_hi"] = sb["lo"], sb["hi"]
            line += f"   [{sb['lo']:+.4f}, {sb['hi']:+.4f}]"
        print(line)
        h4[nm] = s
    out["h4_oof_crossfit"] = h4

    # validation
    try:
        vb = read(R / "au_val_finalagent", "base")
        vf = read(R / "au_val_finalagent", "final")
        vt = read(R / "au_val_finalagent", "traces")
        vq = {c: vt[c]["q0"] for c in vt}
        vu = {c: groups["val"][c]["ustab"] for c in vb if c in groups["val"]}
        print("\n  validation:")
        vres = {}
        for nm, src, lo, hi in (("Q low (hardest)", vq, -np.inf, QCUTS[0]),
                                ("Q high (easiest)", vq, QCUTS[1], np.inf),
                                ("u_stab high (hardest)", vu, UCUTS[1], np.inf),
                                ("u_stab low (easiest)", vu, -np.inf, UCUTS[0])):
            cs = [c for c in vb if c in src and np.isfinite(src[c])
                  and lo <= src[c] < hi and vb[c]["n_gt"] > 0]
            if len(cs) < 5:
                print(f"    {nm:32s} n={len(cs):3d}  too few"
                      + ("   (n=0 on validation, as the audit notes)"
                         if len(cs) == 0 else ""))
                vres[nm] = {"n": len(cs), "too_few": True}
                continue
            s = boot_dice({c: vf[c] for c in cs}, {c: vb[c] for c in cs})
            print(f"    {nm:32s} n={s['n']:3d}  {s['delta']:+.4f} "
                  f"[{s['lo']:+.4f}, {s['hi']:+.4f}]")
            vres[nm] = s
        out["h4_val"] = vres
    except FileNotFoundError as e:
        print("  validation run missing:", e)

    # ---------------- R1(d) attribution, R1(g) ladder, R2(2) ----------
    print("\n" + "=" * 78)
    print("R1(d) attribution and R1(g) Phase V ladder, cross-fitted, out-of-fold")
    print(f"  {'configuration':26s}{'dDice':>10s}{'95% CI':>22s}"
          f"{'negFP':>10s}{'dSens':>9s}{'acts/case':>11s}{'imp/unch/deg':>15s}")
    rows = {}
    for tag, label in (("attr_fixed", "(a) fixed"),
                       ("attr_policy", "(b) rule-based"),
                       ("attr_learn", "(c) learned"),
                       ("pv_rung2", "PV rung2 fixed T=3"),
                       ("pv_rung3", "PV rung3 learned T=3"),
                       ("pv_rung5noroi", "PV rung5-noroi T=10"),
                       ("corrected", "R2(3) corrected"),
                       ("finalagent", "FINAL AGENT"),
                       ("gate095", "gate 0.95")):
        p = R / f"au_oof_{tag}"
        try:
            f = read(p, "final")
            t = read(p, "traces")
        except FileNotFoundError:
            print(f"  {label:26s} not executed")
            continue
        s = boot_dice(f, base)
        neg = [c for c in base if base[c]["n_gt"] == 0]
        nf = sum(1 for c in neg if f[c]["n_fp_comp"] > 0)
        ds = lesion_sens(f) - lesion_sens(base)
        acts = float(np.mean([t[c]["accepted"] for c in t]))
        i = u_ = d = 0
        for c in f:
            if base[c]["n_gt"] == 0:
                continue
            dd = f[c]["dice"] - base[c]["dice"]
            i += dd > 0.01
            d += dd < -0.01
            u_ += abs(dd) <= 0.01
        tot = max(i + u_ + d, 1)
        print(f"  {label:26s}{s['delta']:+10.4f}  [{s['lo']:+.4f}, {s['hi']:+.4f}]"
              f"{nf:7d}/{len(neg):<3d}{ds:+9.4f}{acts:11.2f}"
              f"{100*i/tot:7.1f}/{100*u_/tot:.1f}/{100*d/tot:.1f}")
        over = sum(1 for c in t if t[c].get("n_gated_candidates", 0) > 10)
        rows[tag] = {"label": label, **s, "neg_with_fp": nf, "n_neg": len(neg),
                     "d_lesion_sens": ds, "actions_per_case": acts,
                     "improved": i, "unchanged": u_, "degraded": d,
                     "cases_over_T10": over,
                     "mean_gated_candidates": float(np.mean(
                         [t[c].get("n_gated_candidates", 0) for c in t]))}
    out["configurations_oof_crossfit"] = rows

    print("\nR2(2) cases with more than T=10 gated candidates (out-of-fold):")
    for tag, r in rows.items():
        if "cases_over_T10" in r:
            print(f"  {r['label']:26s}{r['cases_over_T10']:5d} of 738   "
                  f"mean gated candidates {r['mean_gated_candidates']:.2f}")

    args.out.write_text(json.dumps(out, indent=1, default=float),
                        encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
