#!/usr/bin/env python3
"""Experiment 2, case and component level, plus the combined uncertainty.

Reads what the earlier passes produced and answers the questions the plan asks,
fitting everything that needs fitting on val and applying it to test.

Component level (the part Phase III would act on):
  * AUROC of each component's mean and max uncertainty for separating a true
    positive from a false positive, over all patients and over tumour-free
    patients alone -- the latter being the case where a reader has nothing else
    to go on.
  * For every missed true lesion, whether it sits in an uncertain region. A
    miss the model was unsure about is recoverable by lowering a threshold; a
    miss it was confident about is not, and the two call for completely
    different Phase III designs.

Case level:
  * Spearman correlation of case uncertainty against 1 - Dice, false-positive
    volume and HD95.
  * Low / medium / high uncertainty thirds with cut-offs taken from val only.

Combined uncertainty (eq. 10): the three maps z-scored and summed with equal
weights, and with weights fitted by logistic regression on val. Because the
component rows carry per-component summaries rather than voxels, the combination
is formed at component level here, which is the level the decision is made at.

    python scripts/exp2_analysis.py --val-components reports/components_val.csv \\
        --test-components reports/components_test.csv \\
        --val-cases reports/exp2_val_cases.json --test-cases reports/exp2_test_cases.json \\
        --val-eval reports/eval505ens_val_full.csv --test-eval reports/eval505ens_test_full.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

U_KEYS = ("u_ent", "u_epi", "u_stab")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auroc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """AUROC plus a bootstrap standard error, or nan if a class is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    u = float(stats.mannwhitneyu(pos, neg).statistic)
    a = u / (float(len(pos)) * float(len(neg)))
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(200):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        boot.append(float(stats.mannwhitneyu(p, n).statistic) / (len(p) * len(n)))
    return a, float(np.std(boot))


def fit_zscore(rows: list[dict], fields: list[str]) -> dict:
    out = {}
    for f in fields:
        v = np.array([float(r[f]) for r in rows], dtype=np.float64)
        out[f] = (float(v.mean()), float(v.std()) or 1.0)
    return out


def combined(rows: list[dict], fields: list[str], zs: dict,
             weights: np.ndarray | None = None) -> np.ndarray:
    z = np.stack([(np.array([float(r[f]) for r in rows]) - zs[f][0]) / zs[f][1]
                  for f in fields])
    w = np.ones(len(fields)) if weights is None else weights
    return (w[:, None] * z).sum(axis=0)


def component_block(name: str, rows: list[dict], zs: dict, w_fit: np.ndarray | None):
    pred = [r for r in rows if r["kind"] == "pred"]
    tp = np.array([i for i, r in enumerate(pred) if r["status"] == "TP"])
    fp = np.array([i for i, r in enumerate(pred) if r["status"] == "FP"])
    print(f"\n--- {name}: predicted components, TP vs FP "
          f"({len(tp)} TP, {len(fp)} FP) ---")
    # "on tumour-free patients only" cannot mean TP vs FP within those
    # patients: one with no disease has no true positive to compare against,
    # so that comparison is empty by construction. The well-posed version,
    # and the one a reader actually faces, is whether a spurious finding in a
    # disease-free patient looks different from a genuine lesion anywhere --
    # FP-on-tumour-free against all TP.
    print(f"{'score':22s} {'AUROC':>7s} {'+-':>6s}   {'FP(neg case) vs all TP':>24s}")
    neg_set = {i for i, r in enumerate(pred) if r["case_gt_positive"] == "0"}

    def show(label, vals):
        # a high-uncertainty component should be MORE likely to be a false
        # positive, so FP is the positive class here
        a, se = auroc(vals[fp], vals[tp])
        sub_fp = np.array([i for i in fp if i in neg_set], dtype=int)
        a2, _ = (auroc(vals[sub_fp], vals[tp]) if len(sub_fp) and len(tp)
                 else (float("nan"), 0.0))
        print(f"{label:22s} {a:7.4f} {se:6.3f}   {a2:>26.4f}")
        return a

    for k in U_KEYS:
        for agg in ("mean", "max"):
            show(f"{k}_{agg}", np.array([float(r[f"{k}_{agg}"]) for r in pred]))
    show("volume_ml", np.array([float(r["volume_ml"]) for r in pred]))
    show("suv_max", np.array([float(r["suv_max"]) for r in pred]))
    fields = [f"{k}_mean" for k in U_KEYS]
    show("combined U (equal w)", combined(pred, fields, zs))
    if w_fit is not None:
        show("combined U (val-fitted)", combined(pred, fields, zs, w_fit))

    # missed lesions: was the model at least unsure there?
    true_rows = [r for r in rows if r["kind"] == "true"]
    hit = [r for r in true_rows if r["status"] == "hit"]
    miss = [r for r in true_rows if r["status"] == "missed"]
    print(f"\n--- {name}: true lesions, hit {len(hit)} vs missed {len(miss)} ---")
    for k in U_KEYS:
        h = np.array([float(r[f"{k}_mean"]) for r in hit])
        m = np.array([float(r[f"{k}_mean"]) for r in miss])
        print(f"  {k}_mean   hit {h.mean():.5f}  missed {m.mean():.5f}  "
              f"| missed at exactly 0: {(m == 0).mean():.1%}")
    pm = np.array([float(r["p_bar_max"]) for r in miss])
    print(f"  P_bar max inside a missed lesion: median {np.median(pm):.4f}, "
          f"90th pct {np.percentile(pm, 90):.4f}, "
          f"share above 0.1: {(pm > 0.1).mean():.1%}")
    vm = np.array([float(r["volume_ml"]) for r in miss])
    vh = np.array([float(r["volume_ml"]) for r in hit])
    print(f"  volume mL: missed median {np.median(vm):.3f}, hit median {np.median(vh):.3f}")
    sm = np.array([float(r["suv_max"]) for r in miss])
    sh = np.array([float(r["suv_max"]) for r in hit])
    print(f"  SUVmax   : missed median {np.median(sm):.2f}, hit median {np.median(sh):.2f}")
    return fields


def case_block(name: str, cases: list[dict], ev: dict, cuts: dict | None):
    rows = [c for c in cases if c["case"] in ev]
    pos = [c for c in rows if c["gt_positive"] == 1]
    print(f"\n--- {name}: case level ({len(rows)} cases, {len(pos)} tumour-positive) ---")
    print(f"{'case score':24s} {'rho(1-Dice)':>12s} {'rho(FP mL)':>11s} {'rho(HD95)':>10s}")
    targets = {}
    for c in pos:
        e = ev[c["case"]]
        targets.setdefault("dice", []).append(1.0 - float(e["dice"]))
    fp_all = np.array([float(ev[c["case"]]["fp_volume_ml"]) for c in rows])
    hd = np.array([float(ev[c["case"]]["hd95_mm"]) for c in pos])
    hd_ok = np.isfinite(hd)
    one_minus_dice = np.array(targets["dice"])

    best = None
    for k in U_KEYS:
        for agg in ("mean_pred", "sum_pred", "mean_near", "sum_near", "sum_body"):
            key = f"{k}_{agg}"
            v_pos = np.array([float(c[key]) for c in pos])
            v_all = np.array([float(c[key]) for c in rows])
            ok = np.isfinite(v_pos)
            r1 = stats.spearmanr(v_pos[ok], one_minus_dice[ok]).statistic if ok.sum() > 3 else np.nan
            r2 = stats.spearmanr(v_all[np.isfinite(v_all)],
                                 fp_all[np.isfinite(v_all)]).statistic
            ok2 = ok & hd_ok
            r3 = stats.spearmanr(v_pos[ok2], hd[ok2]).statistic if ok2.sum() > 3 else np.nan
            print(f"{key:24s} {r1:12.4f} {r2:11.4f} {r3:10.4f}")
            if best is None or (np.isfinite(r1) and abs(r1) > abs(best[1])):
                best = (key, r1)

    # tertiles on the strongest val score, cut-offs from val only
    key = cuts["key"] if cuts else best[0]
    v_all = np.array([float(c[key]) for c in rows])
    if cuts is None:
        lo, hi = np.nanpercentile(v_all, [33.333, 66.667])
        cuts = {"key": key, "lo": float(lo), "hi": float(hi)}
        print(f"\n  tertile cut-offs fitted on val for '{key}': "
              f"{cuts['lo']:.4g} / {cuts['hi']:.4g}")
    else:
        print(f"\n  tertile cut-offs taken from val for '{key}': "
              f"{cuts['lo']:.4g} / {cuts['hi']:.4g}")
    grp = np.digitize(v_all, [cuts["lo"], cuts["hi"]])
    print(f"  {'group':8s} {'n':>4s} {'n pos':>6s} {'Dice':>7s} {'FP mL':>8s} {'FN mL':>8s}")
    for g, lab in enumerate(("low", "medium", "high")):
        sel = [rows[i] for i in np.flatnonzero(grp == g)]
        if not sel:
            continue
        d = [float(ev[c["case"]]["dice"]) for c in sel if c["gt_positive"] == 1]
        f = [float(ev[c["case"]]["fp_volume_ml"]) for c in sel]
        n = [float(ev[c["case"]]["fn_volume_ml"]) for c in sel]
        print(f"  {lab:8s} {len(sel):>4d} {len(d):>6d} "
              f"{np.mean(d) if d else float('nan'):7.4f} {np.mean(f):8.3f} {np.mean(n):8.3f}")
    return cuts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("val-components", "test-components", "val-cases", "test-cases",
              "val-eval", "test-eval"):
        ap.add_argument(f"--{a}", type=Path, required=True)
    args = ap.parse_args()

    vc = read_csv(args.val_components)
    tc = read_csv(args.test_components)
    fields = [f"{k}_mean" for k in U_KEYS]
    zs = fit_zscore([r for r in vc if r["kind"] == "pred"], fields)   # val only

    # logistic weights, fitted on val predicted components only
    pv = [r for r in vc if r["kind"] == "pred"]
    X = np.stack([(np.array([float(r[f]) for r in pv]) - zs[f][0]) / zs[f][1]
                  for f in fields]).T
    y = np.array([1.0 if r["status"] == "FP" else 0.0 for r in pv])
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(300):                      # plain gradient ascent, no sklearn
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = X.T @ (y - p) / len(y)
        w += 1.0 * g
        b += 1.0 * float((y - p).mean())
    print("val-fitted weights for combined U (FP as positive class):")
    for f, wi in zip(fields, w):
        print(f"   {f:14s} {wi:+.4f}")

    component_block("VAL", vc, zs, w)
    component_block("TEST", tc, zs, w)

    ev_v = {r["case"]: r for r in read_csv(args.val_eval)}
    ev_t = {r["case"]: r for r in read_csv(args.test_eval)}
    cases_v = json.loads(args.val_cases.read_text())
    cases_t = json.loads(args.test_cases.read_text())
    cuts = case_block("VAL", cases_v, ev_v, None)
    case_block("TEST", cases_t, ev_t, cuts)


if __name__ == "__main__":
    main()
