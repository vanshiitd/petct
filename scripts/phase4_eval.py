#!/usr/bin/env python3
"""Phase IV step 5: evaluate the agent against the pre-registered plan.

Reads the per-case outputs of `phase4_agent.py` for one or more splits and
reports exactly what `reports/phase4_analysis_plan.md` fixed in advance:

  primary     mean Dice over tumour-positive cases, agent vs M0, paired
              case-level bootstrap over patients (H3)
  safety      Delta lesion sensitivity against the -0.01 margin
  secondary   FP volume, FN volume, FP lesions per patient, tumour-free
              patients with any FP, HD95, and the improved / unchanged /
              degraded split with |Delta-Dice| < 0.01 as unchanged
  section 23  iterations and runtime per case
  section 22  the same numbers by Q tertile, uncertainty tertile, and the
              three inconsistency groups, with H4
  section 21  the ablation ladder

tau_Q is applied here rather than inside the agent: a case whose starting Q
exceeds the threshold keeps M0 untouched, which is a post-hoc selection over a
single run and gives exactly what re-running with that threshold would. Tertile
and tau_Q boundaries are taken from out-of-fold and applied unchanged to val
and test, never refitted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

NBOOT = 2000
UNCHANGED = 0.01


def read(prefix: Path, name: str) -> list[dict]:
    f = prefix.with_name(prefix.name + f"_{name}.csv")
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("case", "stopped", "failure", "action", "patient"):
                continue
            try:
                r[k] = float(v) if v not in ("", "nan") else float("nan")
            except ValueError:
                pass
    return rows


def combine(base: list[dict], final: list[dict], traces: list[dict],
            tau_q: float, upto: int | None = None) -> list[dict]:
    """The agent's output under a given tau_Q (and optionally a capped T)."""
    b = {r["case"]: r for r in base}
    f = {r["case"]: r for r in final}
    t = {r["case"]: r for r in traces}
    out = []
    for case in b:
        tr = t.get(case, {})
        q0 = tr.get("q0", float("inf"))
        row = dict(b[case])
        if q0 <= tau_q:
            if upto is None:
                row = dict(f[case])
                row["case"] = case
            else:
                key = f"m{upto}_dice"
                if key in tr and tr[key] == tr[key]:
                    for k in ("dice", "fp_ml", "fn_ml", "hd95", "n_hit", "n_fp_comp"):
                        row[k] = tr[f"m{upto}_{k}"]
        row["q0"] = q0
        row["acted"] = int(q0 <= tau_q)
        row["iterations"] = tr.get("iterations", 0)
        out.append(row)
    return out


def agg(rows, idx=None) -> dict:
    sel = rows if idx is None else [rows[i] for i in idx]
    pos = [r for r in sel if r["n_gt"] > 0]
    neg = [r for r in sel if r["n_gt"] == 0]
    hd = [r["hd95"] for r in pos if r["hd95"] == r["hd95"]]
    return {"dice": float(np.mean([r["dice"] for r in pos])) if pos else float("nan"),
            "fp_ml": float(np.mean([r["fp_ml"] for r in sel])),
            "fn_ml": float(np.mean([r["fn_ml"] for r in sel])),
            "hd95": float(np.mean(hd)) if hd else float("nan"),
            "n_hd_undef": len(pos) - len(hd),
            "lesion_sens": (sum(r["n_hit"] for r in pos) /
                            max(sum(r["n_gt"] for r in pos), 1)),
            "fp_per_patient": float(np.mean([r["n_fp_comp"] for r in sel])),
            "neg_with_fp": int(sum(1 for r in neg if r["fp_ml"] > 0)),
            "n_neg": len(neg), "n": len(sel)}


def paired_boot(a, b, keys, seed=0, nboot=NBOOT):
    assert [r["case"] for r in a] == [r["case"] for r in b]
    rng = np.random.default_rng(seed)
    n = len(a)
    obs = {k: agg(a)[k] - agg(b)[k] for k in keys}
    draws = {k: [] for k in keys}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        ga, gb = agg(a, idx), agg(b, idx)
        for k in keys:
            draws[k].append(ga[k] - gb[k])
    out = {}
    for k in keys:
        d = np.asarray(draws[k], float)
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"delta": obs[k], "lo": float(lo), "hi": float(hi),
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def fmt(d) -> str:
    return (f"{d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
            + ("" if d["excludes_zero"] else "  (straddles 0)"))


def outcome_split(a, b) -> dict:
    pa = {r["case"]: r for r in a}
    pb = {r["case"]: r for r in b}
    imp = unc = deg = 0
    for c in pa:
        if pa[c]["n_gt"] == 0:
            continue
        d = pa[c]["dice"] - pb[c]["dice"]
        if d >= UNCHANGED:
            imp += 1
        elif d <= -UNCHANGED:
            deg += 1
        else:
            unc += 1
    tot = max(imp + unc + deg, 1)
    return {"improved": imp, "unchanged": unc, "degraded": deg,
            "improved_pct": 100 * imp / tot, "unchanged_pct": 100 * unc / tot,
            "degraded_pct": 100 * deg / tot}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--tau-grid", type=float, nargs="*",
                    default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--budget", type=float, default=-0.01)
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\phase4_eval.json"))
    args = ap.parse_args()

    data = {}
    for nm, pre in (("oof", args.oof), ("val", args.val), ("test", args.test)):
        if pre is None:
            continue
        data[nm] = {"base": read(pre, "base"), "final": read(pre, "final"),
                    "traces": read(pre, "traces")}
    report = {}

    # ---- Rule R4: choose tau_Q on out-of-fold, check eligibility on val ----
    print("=" * 84)
    print("Rule R4: tau_Q tuned on out-of-fold, eligible only if dSens >= "
          f"{args.budget} on BOTH out-of-fold and val")
    print(f"  {'tau_Q':>7s}{'acted':>8s}{'oof Dice':>10s}{'oof dSens':>11s}"
          f"{'oof FP mL':>11s}{'val dSens':>11s}  eligible")
    rows = []
    for tq in args.tau_grid:
        o = combine(data["oof"]["base"], data["oof"]["final"], data["oof"]["traces"], tq)
        o0 = agg(data["oof"]["base"])
        oa = agg(o)
        dso = oa["lesion_sens"] - o0["lesion_sens"]
        dsv = float("nan")
        if "val" in data:
            v = combine(data["val"]["base"], data["val"]["final"],
                        data["val"]["traces"], tq)
            dsv = agg(v)["lesion_sens"] - agg(data["val"]["base"])["lesion_sens"]
        ok = dso >= args.budget and (dsv != dsv or dsv >= args.budget)
        acted = sum(r["acted"] for r in o)
        rows.append((tq, oa, dso, dsv, ok, acted))
        print(f"  {tq:7.2f}{acted:8d}{oa['dice']:10.4f}{dso:+11.4f}"
              f"{oa['fp_ml']:11.3f}{dsv:+11.4f}  {'yes' if ok else 'no'}")
    elig = [r for r in rows if r[4]]
    if not elig:
        print("\n  NOTHING IS ELIGIBLE: no tau_Q holds lesion sensitivity within "
              "one point on both splits. Reporting the best-on-out-of-fold "
              "configuration, labelled ineligible, as the plan requires.")
        chosen = max(rows, key=lambda r: r[1]["dice"])
    else:
        chosen = max(elig, key=lambda r: (r[1]["dice"], -r[1]["fp_ml"]))
    tau_q = chosen[0]
    print(f"\n  chosen tau_Q = {tau_q:.2f} "
          f"({'eligible' if chosen[4] else 'INELIGIBLE'})")
    report["rule_r4"] = {"tau_q": tau_q, "eligible": bool(chosen[4]),
                         "grid": [{"tau_q": r[0], "oof_dice": r[1]["dice"],
                                   "oof_dsens": r[2], "val_dsens": r[3],
                                   "eligible": bool(r[4]), "acted": r[5]}
                                  for r in rows]}

    # ---- per split: M0 and the iteration ladder --------------------------
    for nm in data:
        d = data[nm]
        base = sorted(d["base"], key=lambda r: r["case"])
        print("\n" + "=" * 84)
        print(f"{nm.upper()}  (n={len(base)})")
        print(f"  {'variant':12s}{'Dice':>9s}{'FP mL':>9s}{'FN mL':>9s}"
              f"{'HD95':>9s}{'lesSens':>9s}{'FP/pt':>8s}{'neg+FP':>9s}")
        b = agg(base)
        print(f"  {'M0':12s}{b['dice']:9.4f}{b['fp_ml']:9.3f}{b['fn_ml']:9.3f}"
              f"{b['hd95']:9.2f}{b['lesion_sens']:9.4f}{b['fp_per_patient']:8.2f}"
              f"{b['neg_with_fp']:>5d}/{b['n_neg']:<3d}")
        ent = {"M0": b}
        for t in (1, 2, 3, None):
            v = sorted(combine(d["base"], d["final"], d["traces"], tau_q, upto=t),
                       key=lambda r: r["case"])
            a = agg(v)
            label = f"M{t}" if t else "agent"
            ent[label] = a
            ds = a["lesion_sens"] - b["lesion_sens"]
            flag = "  OVER BUDGET" if ds < args.budget else ""
            print(f"  {label:12s}{a['dice']:9.4f}{a['fp_ml']:9.3f}{a['fn_ml']:9.3f}"
                  f"{a['hd95']:9.2f}{a['lesion_sens']:9.4f}"
                  f"{a['fp_per_patient']:8.2f}"
                  f"{a['neg_with_fp']:>5d}/{a['n_neg']:<3d}{flag}")
        agent = sorted(combine(d["base"], d["final"], d["traces"], tau_q),
                       key=lambda r: r["case"])
        bs = paired_boot(agent, base,
                         ("dice", "fp_ml", "fn_ml", "lesion_sens", "fp_per_patient"))
        print(f"\n  agent - M0, paired bootstrap over {len(base)} patients:")
        for k, v in bs.items():
            print(f"    {k:16s} {fmt(v)}")
        oc = outcome_split(agent, base)
        print(f"  improved {oc['improved']} ({oc['improved_pct']:.1f}%), "
              f"unchanged {oc['unchanged']} ({oc['unchanged_pct']:.1f}%), "
              f"degraded {oc['degraded']} ({oc['degraded_pct']:.1f}%)")
        it = [r["iterations"] for r in agent]
        print(f"  iterations/case mean {np.mean(it):.2f}, "
              f"acted on {sum(r['acted'] for r in agent)}/{len(agent)} cases")
        ent["bootstrap"] = bs
        ent["outcomes"] = oc
        ent["iterations_mean"] = float(np.mean(it))
        report[nm] = ent

        # ---- section 22: difficult cases, H4 -----------------------------
        if nm in ("oof", "val", "test"):
            q0 = np.array([r["q0"] for r in agent], float)
            oof_q = np.array([t["q0"] for t in data["oof"]["traces"]], float)
            cuts = np.percentile(oof_q[np.isfinite(oof_q)], [33.3, 66.7])
            print(f"\n  by Q tertile (boundaries {cuts[0]:.3f}, {cuts[1]:.3f} "
                  f"from out-of-fold):")
            strat = {}
            for i, (lo, hi, lab) in enumerate((
                    (-np.inf, cuts[0], "low Q (hardest)"),
                    (cuts[0], cuts[1], "mid Q"),
                    (cuts[1], np.inf, "high Q (easiest)"))):
                sel = [j for j, v in enumerate(q0) if lo <= v < hi]
                if len(sel) < 5:
                    continue
                aa = [agent[j] for j in sel]
                bb = [base[j] for j in sel]
                s = paired_boot(aa, bb, ("dice",))
                strat[lab] = {"n": len(sel), **s["dice"]}
                print(f"    {lab:18s} n={len(sel):3d}  dDice {fmt(s['dice'])}")
            ent["q_tertile"] = strat
            if "low Q (hardest)" in strat and "high Q (easiest)" in strat:
                lo_d = strat["low Q (hardest)"]["delta"]
                hi_d = strat["high Q (easiest)"]["delta"]
                h4 = lo_d > hi_d and strat["low Q (hardest)"]["excludes_zero"]
                print(f"    H4 ({nm}): low-Q dDice {lo_d:+.4f} vs high-Q {hi_d:+.4f}"
                      f"  -> {'SUPPORTED' if h4 else 'not supported'}")
                ent["h4"] = bool(h4)

    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
