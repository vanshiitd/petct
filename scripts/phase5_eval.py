#!/usr/bin/env python3
"""Phase V evaluation, exactly as `reports/phase5_analysis_plan.md` fixed it.

Two co-primary endpoints, which Phase IV did not have:

  1  mean Dice over tumour-positive cases, paired case-level bootstrap over
     patients (2000 resamples)
  2  the number of tumour-free patients carrying at least one false positive,
     paired by patient with an exact two-sided McNemar test, with FP volume
     over all cases as its supporting measure

Success requires non-inferiority on both (Dice margin -0.01, FP-count margin 0
patients) AND superiority on at least one, with the two superiority tests
corrected by Holm at family-wise alpha 0.05.

The lesion-sensitivity safety constraint (margin -0.01) overrides both: fail it
and the phase is a failure whatever the co-primaries show.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

NBOOT = 2000
ALPHA = 0.05
DICE_MARGIN = -0.01
SENS_MARGIN = -0.01


def read(prefix: Path, name: str) -> dict:
    f = prefix.with_name(prefix.name + f"_{name}.csv")
    out = {}
    for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
        for k, v in list(r.items()):
            if k == "case":
                continue
            try:
                r[k] = float(v) if v not in ("", "nan") else float("nan")
            except ValueError:
                pass
        out[r["case"]] = r
    return out


def paired_dice(a: dict, b: dict, seed: int = 0):
    """Mean Dice over tumour-positive cases, a - b, bootstrapped over patients."""
    cases = sorted(set(a) & set(b))
    pos = [c for c in cases if b[c]["n_gt"] > 0]
    da = np.array([a[c]["dice"] for c in pos], float)
    db = np.array([b[c]["dice"] for c in pos], float)
    delta = float(np.nanmean(da) - np.nanmean(db))
    rng = np.random.default_rng(seed)
    n = len(pos)
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, n, n)
        boot[i] = np.nanmean(da[ix]) - np.nanmean(db[ix])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p for superiority (H0: delta <= 0)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"n": n, "a": float(np.nanmean(da)), "b": float(np.nanmean(db)),
            "delta": delta, "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, max(p, 1.0 / NBOOT)))}


def mcnemar_fp(a: dict, b: dict):
    """Tumour-free patients with any FP, paired, exact two-sided McNemar."""
    cases = sorted(set(a) & set(b))
    neg = [c for c in cases if b[c]["n_gt"] == 0]
    fa = np.array([a[c]["n_fp_comp"] > 0 for c in neg])
    fb = np.array([b[c]["n_fp_comp"] > 0 for c in neg])
    b01 = int((~fa & fb).sum())      # cleared by a
    b10 = int((fa & ~fb).sum())      # newly dirty under a
    n_disc = b01 + b10
    p = (binomtest(b10, n_disc, 0.5).pvalue if n_disc else 1.0)
    return {"n_neg": len(neg), "a": int(fa.sum()), "b": int(fb.sum()),
            "cleared": b01, "added": b10, "discordant": n_disc,
            "p": float(p)}


def paired_mean(a: dict, b: dict, key: str, only=None, seed: int = 0):
    cases = sorted(set(a) & set(b))
    if only == "pos":
        cases = [c for c in cases if b[c]["n_gt"] > 0]
    da = np.array([a[c][key] for c in cases], float)
    db = np.array([b[c][key] for c in cases], float)
    rng = np.random.default_rng(seed)
    n = len(cases)
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, n, n)
        boot[i] = np.nanmean(da[ix]) - np.nanmean(db[ix])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(np.nanmean(da) - np.nanmean(db)),
            "lo": float(lo), "hi": float(hi),
            "a": float(np.nanmean(da)), "b": float(np.nanmean(db))}


def lesion_sens(d: dict):
    hit = sum(r["n_hit"] for r in d.values())
    gt = sum(r["n_gt"] for r in d.values())
    return hit / gt if gt else float("nan")


def holm(pvals: dict, alpha: float = ALPHA) -> dict:
    """Holm step-down. Returns {name: (p, threshold, rejected)}."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    out, still = {}, True
    for i, (name, p) in enumerate(order):
        thr = alpha / (m - i)
        rej = still and p <= thr
        still = rej
        out[name] = {"p": p, "threshold": thr, "rejected": bool(rej)}
    return out


def evaluate(agent_prefix: Path, ref_prefix: Path, label: str) -> dict:
    a = read(agent_prefix, "final")
    b = read(ref_prefix, "base")
    common = sorted(set(a) & set(b))
    a = {c: a[c] for c in common}
    b = {c: b[c] for c in common}

    dice = paired_dice(a, b)
    fp = mcnemar_fp(a, b)
    sens_a, sens_b = lesion_sens(a), lesion_sens(b)
    d_sens = sens_a - sens_b

    print(f"\n{'='*76}\n{label}  (n={len(common)})")
    print(f"  co-primary 1  Dice over {dice['n']} tumour-positive cases")
    print(f"      {dice['b']:.4f} -> {dice['a']:.4f}   "
          f"delta {dice['delta']:+.4f} [{dice['lo']:+.4f}, {dice['hi']:+.4f}]"
          f"   p={dice['p']:.4f}")
    ni_dice = dice["lo"] > DICE_MARGIN
    print(f"      non-inferior (lower bound > {DICE_MARGIN}): "
          f"{'YES' if ni_dice else 'NO'}")

    print(f"  co-primary 2  tumour-free patients with any FP "
          f"(n={fp['n_neg']}), exact McNemar")
    print(f"      {fp['b']} -> {fp['a']}   cleared {fp['cleared']}, "
          f"added {fp['added']}, discordant {fp['discordant']}   "
          f"p={fp['p']:.4g}")
    ni_fp = fp["a"] <= fp["b"]
    print(f"      non-inferior (count not increased): "
          f"{'YES' if ni_fp else 'NO'}")

    hres = holm({"dice": dice["p"], "fp_count": fp["p"]})
    print(f"  Holm at family-wise alpha {ALPHA}:")
    for k, v in sorted(hres.items(), key=lambda kv: kv[1]["p"]):
        print(f"      {k:10s} p={v['p']:.4g}  threshold {v['threshold']:.4f}"
              f"  {'REJECTED H0' if v['rejected'] else 'not rejected'}")

    sup_dice = hres["dice"]["rejected"] and dice["delta"] > 0
    sup_fp = hres["fp_count"]["rejected"] and fp["a"] < fp["b"]
    ok_safety = d_sens >= SENS_MARGIN
    success = ni_dice and ni_fp and (sup_dice or sup_fp) and ok_safety

    print(f"  safety  lesion sensitivity {sens_b:.4f} -> {sens_a:.4f}  "
          f"delta {d_sens:+.4f}  "
          f"{'within' if ok_safety else 'OVER'} the {SENS_MARGIN} margin")
    print(f"  VERDICT: {'SUCCESS' if success else 'not successful'}"
          f"  (non-inf both={ni_dice and ni_fp}, superior>=1="
          f"{sup_dice or sup_fp}, safety={ok_safety})")

    for k, only, lab in (("fp_ml", None, "FP volume mL (all cases)"),
                         ("fn_ml", None, "FN volume mL (all cases)"),
                         ("hd95", "pos", "HD95 (tumour-positive)")):
        m = paired_mean(a, b, k, only)
        print(f"  {lab:32s} {m['b']:8.3f} -> {m['a']:8.3f}  "
              f"{m['delta']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]")

    return {"label": label, "dice": dice, "fp_count": fp, "holm": hres,
            "lesion_sens": {"a": sens_a, "b": sens_b, "delta": d_sens,
                            "within_margin": bool(ok_safety)},
            "success": bool(success)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=Path, nargs="+", required=True,
                    help="one or more run prefixes to evaluate")
    ap.add_argument("--ref", type=Path, required=True,
                    help="prefix whose _base.csv holds M0 for these cases")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\phase5_eval.json"))
    args = ap.parse_args()

    labels = args.labels or [p.name for p in args.agent]
    res = [evaluate(p, args.ref, lab) for p, lab in zip(args.agent, labels)]
    args.out.write_text(json.dumps(res, indent=1, default=float),
                        encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
