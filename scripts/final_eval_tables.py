#!/usr/bin/env python3
"""Work plan sections 21-23: the final descriptive tables.

Everything here is descriptive. The project's confirmatory result is Phase V's
single test pass; this reports the frozen configurations of
`reports/final_eval_plan.md` across out-of-fold, validation and test so the
thesis has one place that shows what each component contributes.

Scoring goes through the same helpers Phase V used (`phase5_eval`), so no row in
any table comes from a different metric implementation than any other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice, mcnemar_fp, paired_mean, lesion_sens

R = Path(r"D:\data\reports")

RUNGS = [
    ("M0", None, "Baseline (5-fold ensemble)"),
    ("unc", "fe_{sp}_unc", "+ uncertainty (gate 0.90, frozen)"),
    ("uncmatch", "fe_{sp}_uncmatch", "+ uncertainty (matched rate, added)"),
    ("reason", "fe_{sp}_reason", "+ reasoning"),
    ("refine", "fe_{sp}_refine", "+ refinement (T=1, no Q gate)"),
    ("reflect", "fe_{sp}_reflect", "+ reflection (Q gate)"),
    ("iterate", "fe_{sp}_iterate", "+ iteration (T=10)"),
    ("learned", "fe_{sp}_learned", "+ learned strategy memory"),
    ("finalagent", "fe_{sp}_finalagent", "FINAL AGENT (fixed strategy, T=10, gated)"),
]


def outcome_split(a: dict, b: dict) -> tuple[int, int, int]:
    imp = unch = deg = 0
    for c in a:
        if b[c]["n_gt"] == 0:
            continue
        d = a[c]["dice"] - b[c]["dice"]
        if d > 0.01:
            imp += 1
        elif d < -0.01:
            deg += 1
        else:
            unch += 1
    return imp, unch, deg


def load_split(sp: str):
    """Every rung for one split, plus the shared M0 baseline."""
    base = read(R / f"fe_{sp}_reason", "base")
    out = {"M0": base}
    for tag, pat, _ in RUNGS:
        if pat is None:
            continue
        f = R / pat.format(sp=sp)
        try:
            out[tag] = read(f, "final")
        except FileNotFoundError:
            out[tag] = None
    return base, out


def section21(splits) -> dict:
    res = {}
    for sp in splits:
        base, runs = load_split(sp)
        print(f"\n{'='*100}\nSECTION 21 ABLATION - {sp.upper()}"
              f"{'   (DESCRIPTIVE)' if sp == 'test' else ''}  n={len(base)}")
        print(f"  {'rung':44s}{'Dice':>8s}{'negFP':>8s}{'FP mL':>8s}"
              f"{'FN mL':>8s}{'lesSens':>9s}{'HD95':>8s}{'imp/unch/deg':>16s}")
        res[sp] = {}
        for tag, pat, label in RUNGS:
            d = runs.get(tag)
            if d is None:
                print(f"  {label:44s}{'not executed':>8s}")
                continue
            pos = [c for c in d if base[c]["n_gt"] > 0]
            neg = [c for c in d if base[c]["n_gt"] == 0]
            dice = float(np.nanmean([d[c]["dice"] for c in pos]))
            negfp = sum(1 for c in neg if d[c]["n_fp_comp"] > 0)
            fp = float(np.nanmean([d[c]["fp_ml"] for c in d]))
            fn = float(np.nanmean([d[c]["fn_ml"] for c in d]))
            hd = float(np.nanmean([d[c]["hd95"] for c in pos]))
            ls = lesion_sens(d)
            i, u, g = outcome_split(d, base)
            tot = max(i + u + g, 1)
            print(f"  {label:44s}{dice:8.4f}{negfp:5d}/{len(neg):<3d}{fp:8.3f}"
                  f"{fn:8.3f}{ls:9.4f}{hd:8.2f}"
                  f"{100*i/tot:6.1f}/{100*u/tot:.1f}/{100*g/tot:.1f}")
            res[sp][tag] = {"dice": dice, "neg_with_fp": negfp, "n_neg": len(neg),
                            "fp_ml": fp, "fn_ml": fn, "lesion_sens": ls,
                            "hd95": hd, "improved": i, "unchanged": u,
                            "degraded": g}
    return res


def tertiles(base: dict, oof_base: dict, oof_vals: dict, key):
    """Boundaries from out-of-fold, applied unchanged to any split."""
    ov = np.array([oof_vals[c] for c in sorted(oof_base)], float)
    cuts = np.nanpercentile(ov[np.isfinite(ov)], [33.3, 66.7])
    return cuts


def section22(splits, diag) -> dict:
    """Difficult cases and H4, baseline against the final agent."""
    res = {}
    # out-of-fold q0 and uncertainty, for boundaries
    oof_tr = read(R / "fe_oof_finalagent", "traces")
    oof_q = {c: oof_tr[c]["q0"] for c in oof_tr}
    qcuts = np.nanpercentile(
        np.array([v for v in oof_q.values()], float), [33.3, 66.7])
    ucuts = np.nanpercentile(
        np.array([diag["oof"][c]["ustab"] for c in diag["oof"]
                  if np.isfinite(diag["oof"][c]["ustab"])], float), [33.3, 66.7])
    print(f"\n{'='*100}\nSECTION 22 DIFFICULT CASES (boundaries from out-of-fold:"
          f" Q {qcuts[0]:.3f}/{qcuts[1]:.3f}, u_stab {ucuts[0]:.4f}/{ucuts[1]:.4f})")

    for sp in splits:
        base = read(R / f"fe_{sp}_reason", "base")
        fin = read(R / f"fe_{sp}_finalagent", "final")
        tr = read(R / f"fe_{sp}_finalagent", "traces")
        res[sp] = {}
        print(f"\n-- {sp.upper()}{'  (DESCRIPTIVE)' if sp == 'test' else ''} --")

        groups = {}
        q = {c: tr[c]["q0"] for c in tr}
        u = {c: diag[sp][c]["ustab"] for c in base if c in diag[sp]}
        for lab, src, cuts in (("Q", q, qcuts), ("u_stab", u, ucuts)):
            for name, lo, hi in (("low (hardest)", -np.inf, cuts[0]),
                                 ("mid", cuts[0], cuts[1]),
                                 ("high (easiest)", cuts[1], np.inf)):
                sel = [c for c in base if c in src
                       and np.isfinite(src[c]) and lo <= src[c] < hi]
                if lab == "u_stab":      # high uncertainty = hardest
                    name = name.replace("low (hardest)", "low (easiest)")
                    name = name.replace("high (easiest)", "high (hardest)")
                groups[f"{lab} {name}"] = sel
        for name, key in (("anatomically inconsistent", "anat"),
                          ("metabolically inconsistent", "metab"),
                          ("topologically inconsistent", "topo")):
            groups[name] = [c for c in base
                            if c in diag[sp] and diag[sp][c][key]]

        for name, sel in groups.items():
            pos = [c for c in sel if base[c]["n_gt"] > 0]
            if len(pos) < 5:
                print(f"  {name:34s} n={len(pos):4d}  too few")
                continue
            a = {c: fin[c] for c in pos}
            b = {c: base[c] for c in pos}
            st = paired_dice(a, b)
            print(f"  {name:34s} n={st['n']:4d}  dDice {st['delta']:+.4f} "
                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}]"
                  f"{'' if st['lo'] > 0 or st['hi'] < 0 else '  (straddles 0)'}")
            res[sp][name] = st
    return res


def section23(splits) -> dict:
    res = {}
    print(f"\n{'='*100}\nSECTION 23 FAILURE RECOVERY - final agent")
    print(f"  {'split':10s}{'improved':>10s}{'unchanged':>11s}{'degraded':>10s}"
          f"{'mean dDice':>12s}{'mean dHD95':>12s}{'iters/case':>12s}{'s/case':>9s}")
    for sp in splits:
        base = read(R / f"fe_{sp}_reason", "base")
        fin = read(R / f"fe_{sp}_finalagent", "final")
        tr = read(R / f"fe_{sp}_finalagent", "traces")
        i, u, g = outcome_split(fin, base)
        tot = max(i + u + g, 1)
        pos = [c for c in fin if base[c]["n_gt"] > 0]
        dd = float(np.nanmean([fin[c]["dice"] - base[c]["dice"] for c in pos]))
        dh = float(np.nanmean([fin[c]["hd95"] - base[c]["hd95"] for c in pos]))
        it = float(np.nanmean([tr[c]["iterations"] for c in tr]))
        sec = float(np.nanmean([tr[c]["seconds"] for c in tr]))
        print(f"  {sp:10s}{100*i/tot:9.1f}%{100*u/tot:10.1f}%{100*g/tot:9.1f}%"
              f"{dd:+12.5f}{dh:+12.3f}{it:12.2f}{sec:9.1f}")
        res[sp] = {"improved_pct": 100*i/tot, "unchanged_pct": 100*u/tot,
                   "degraded_pct": 100*g/tot, "mean_d_dice": dd,
                   "mean_d_hd95": dh, "iters": it, "seconds": sec}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["oof", "val", "test"])
    ap.add_argument("--diag", type=Path,
                    default=Path(r"D:\data\reports\final_case_groups.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\final_eval_tables.json"))
    args = ap.parse_args()

    diag = json.load(open(args.diag, encoding="utf-8"))
    out = {"section21": section21(args.splits),
           "section22": section22(args.splits, diag),
           "section23": section23(args.splits)}
    args.out.write_text(json.dumps(out, indent=1, default=float),
                        encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
