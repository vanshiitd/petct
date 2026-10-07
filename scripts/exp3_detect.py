#!/usr/bin/env python3
"""Experiment 3: does reasoning detect failures better than confidence alone?

Five detectors, run on two tasks:

  task "pred"    -- is this predicted component a false positive?
  task "recover" -- is this sub-threshold candidate a real missed lesion?

  (a) conf    confidence only: mean/max P and mean/max U_ent
  (b) unc     (a) plus flip instability, mean/max U_stab_single
  (c) reason  no model scores at all: metabolic, anatomy, CT, shape, position
  (d) all     (b) + (c)
  (e) rules   a hand-written constraint set with thresholds taken from val

Everything is fitted on val with patient-grouped cross-validation -- folds split
on patient, never on component, because one patient contributes many correlated
components and a random split would leak. The model is then refitted on all of
val and applied once to test.

The fitting set is small: 709 val components for "pred" and, far worse, 312
candidates with only 29 positives for "recover". Logistic regression is kept
standardised and L2-regularised for that reason, and coefficient stability
across folds is reported so an unstable fit is visible rather than implied.
A gradient-boosted tree is run alongside as an upper reference, not as the
headline, because with 29 positives it cannot be trusted.

Missing values (a patient whose field of view excludes the liver, so there is
no reference SUV) are imputed with the val median and flagged by count.

    python scripts/exp3_detect.py --val reports/cand_val.csv --test reports/cand_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

CONF = ["p_mean", "p_max", "uent_mean", "uent_max"]
UNC = CONF + ["ustab_mean", "ustab_max"]
REASON = ["suv_max", "suv_mean", "suv_peak", "suv_max_over_liver",
          "frac_physio", "frac_liver", "frac_spleen", "dist_physio_mm",
          "hu_mean", "frac_fat_hu", "frac_fluid_hu",
          "log_volume", "sphericity", "elongation", "z_rel"]
GROUPS = {"a conf": CONF, "b unc": UNC, "c reason": REASON, "d all": UNC + REASON}


def load(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["kind"] == kind]
    pos = {"pred": "FP", "recover": "recoverable"}[kind]
    y = np.array([1 if r["status"] == pos else 0 for r in rows])
    g = np.array([r["case"] for r in rows])
    return rows, y, g, rows


def matrix(rows: list[dict], feats: list[str], med: dict | None):
    X = np.zeros((len(rows), len(feats)))
    for j, f in enumerate(feats):
        if f == "log_volume":
            v = np.log10(np.array([float(r["volume_ml"]) for r in rows]) + 1e-4)
        else:
            v = np.array([float(r[f]) if r[f] not in ("", "nan") else np.nan
                          for r in rows])
        X[:, j] = v
    if med is None:
        med = {f: float(np.nanmedian(X[:, j])) for j, f in enumerate(feats)}
    n_missing = 0
    for j, f in enumerate(feats):
        m = ~np.isfinite(X[:, j])
        n_missing += int(m.sum())
        X[m, j] = med[f]
    return X, med, n_missing


def rule_score(rows: list[dict], thr: dict) -> np.ndarray:
    """Interpretable constraints; each fired rule adds one point."""
    s = np.zeros(len(rows))
    for i, r in enumerate(rows):
        vol = float(r["volume_ml"])
        sol = float(r["suv_max_over_liver"]) if r["suv_max_over_liver"] not in ("", "nan") \
            else np.nan
        fp_ = float(r["frac_physio"]) if r["frac_physio"] not in ("", "nan") else 0.0
        fat = float(r["frac_fat_hu"]) if r["frac_fat_hu"] not in ("", "nan") else 0.0
        flu = float(r["frac_fluid_hu"]) if r["frac_fluid_hu"] not in ("", "nan") else 0.0
        if vol < thr["vol"] and (not np.isfinite(sol) or sol < thr["suv"]):
            s[i] += 1                                   # small and not hot
        if fp_ > thr["physio"]:
            s[i] += 1                                   # mostly in an uptake organ
        if fat > thr["fat"]:
            s[i] += 1                                   # brown-fat HU range
        if flu > thr["fluid"]:
            s[i] += 1                                   # urine HU range
    return s


def tune_rules(rows, y, groups) -> dict:
    best, best_auc = None, -1.0
    for vol in (0.5, 1.0, 2.0, 5.0):
        for suv in (1.0, 1.5, 2.0, 3.0):
            for ph in (0.3, 0.5, 0.7):
                for fat in (0.3, 0.5, 0.8):
                    for flu in (0.3, 0.5, 0.8):
                        t = {"vol": vol, "suv": suv, "physio": ph,
                             "fat": fat, "fluid": flu}
                        s = rule_score(rows, t)
                        if len(np.unique(s)) < 2:
                            continue
                        a = roc_auc_score(y, s)
                        if a > best_auc:
                            best_auc, best = a, t
    return best


def cv_eval(X, y, groups, clf_factory, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    aucs, aps, coefs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        clf = clf_factory()
        clf.fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        if hasattr(clf, "coef_"):
            coefs.append(clf.coef_[0])
    return np.array(aucs), np.array(aps), (np.array(coefs) if coefs else None)


def paired_bootstrap(y, pa, pd_, n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    base = roc_auc_score(y, pd_) - roc_auc_score(y, pa)
    diffs = []
    for _ in range(n // 10):            # 1000 resamples: AUROC is slow to recompute
        ip = rng.choice(idx_pos, len(idx_pos), replace=True)
        inn = rng.choice(idx_neg, len(idx_neg), replace=True)
        k = np.concatenate([ip, inn])
        yy = y[k]
        diffs.append(roc_auc_score(yy, pd_[k]) - roc_auc_score(yy, pa[k]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, float(lo), float(hi)


def run_task(kind: str, val_path: Path, test_path: Path, out: dict) -> None:
    rows_v, yv, gv, _ = load(val_path, kind)
    rows_t, yt, gt_, _ = load(test_path, kind)
    print(f"\n{'='*78}\nTASK '{kind}': val {len(yv)} candidates ({yv.sum()} positive, "
          f"{yv.mean():.1%}) | test {len(yt)} ({yt.sum()} positive, {yt.mean():.1%})")

    lr = lambda: LogisticRegression(max_iter=5000, C=1.0)
    gb = lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                                learning_rate=0.08, random_state=0)
    res, test_scores = {}, {}
    print(f"\n{'detector':12s} {'val CV AUROC':>18s} {'val CV AP':>16s} "
          f"{'test AUROC':>11s} {'test AP':>9s}")
    print("-" * 72)
    for name, feats in GROUPS.items():
        Xv, med, nm_v = matrix(rows_v, feats, None)
        Xt, _, nm_t = matrix(rows_t, feats, med)
        aucs, aps, coefs = cv_eval(Xv, yv, gv, lr)
        mu, sd = Xv.mean(0), Xv.std(0); sd[sd == 0] = 1.0
        clf = lr(); clf.fit((Xv - mu) / sd, yv)
        pt = clf.predict_proba((Xt - mu) / sd)[:, 1]
        test_scores[name] = pt
        ta, tp_ = roc_auc_score(yt, pt), average_precision_score(yt, pt)
        print(f"{name:12s} {aucs.mean():>9.4f} +-{aucs.std():.4f} "
              f"{aps.mean():>9.4f} +-{aps.std():.4f} {ta:>11.4f} {tp_:>9.4f}")
        res[name] = {"cv_auroc_mean": float(aucs.mean()), "cv_auroc_std": float(aucs.std()),
                     "cv_ap_mean": float(aps.mean()), "test_auroc": float(ta),
                     "test_ap": float(tp_), "n_missing_val": nm_v, "n_missing_test": nm_t,
                     "features": feats,
                     "coef": dict(zip(feats, clf.coef_[0].round(4).tolist())),
                     "coef_cv_std": (dict(zip(feats, coefs.std(0).round(4).tolist()))
                                     if coefs is not None else None)}
    # (e) rules
    thr = tune_rules(rows_v, yv, gv)
    sv, st = rule_score(rows_v, thr), rule_score(rows_t, thr)
    print(f"{'e rules':12s} {roc_auc_score(yv, sv):>9.4f} {'(val fit)':>9s} "
          f"{average_precision_score(yv, sv):>9.4f}{'':>8s} "
          f"{roc_auc_score(yt, st):>11.4f} {average_precision_score(yt, st):>9.4f}")
    print(f"             thresholds from val: {thr}")
    res["e rules"] = {"thresholds": thr, "val_auroc": float(roc_auc_score(yv, sv)),
                      "test_auroc": float(roc_auc_score(yt, st)),
                      "test_ap": float(average_precision_score(yt, st))}
    test_scores["e rules"] = st

    # gradient boosting as an upper reference on the full feature set
    feats = GROUPS["d all"]
    Xv, med, _ = matrix(rows_v, feats, None)
    Xt, _, _ = matrix(rows_t, feats, med)
    aucs, aps, _ = cv_eval(Xv, yv, gv, gb)
    g = gb(); g.fit(Xv, yv)
    ptg = g.predict_proba(Xt)[:, 1]
    print(f"{'d all (GBT)':12s} {aucs.mean():>9.4f} +-{aucs.std():.4f} "
          f"{aps.mean():>9.4f} +-{aps.std():.4f} "
          f"{roc_auc_score(yt, ptg):>11.4f} {average_precision_score(yt, ptg):>9.4f}")
    res["d all (GBT)"] = {"cv_auroc_mean": float(aucs.mean()),
                          "test_auroc": float(roc_auc_score(yt, ptg)),
                          "test_ap": float(average_precision_score(yt, ptg))}

    d, lo, hi = paired_bootstrap(yt, test_scores["a conf"], test_scores["d all"])
    print(f"\npaired bootstrap on test, (d all) - (a conf): {d:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] "
          f"{'excludes 0' if (lo > 0) == (hi > 0) else 'straddles 0'}")
    res["bootstrap_d_vs_a"] = {"delta_auroc": float(d), "ci_lo": lo, "ci_hi": hi}

    print(f"\nlogistic coefficients, detector 'd all' (standardised; "
          f"+ means more likely {'FP' if kind == 'pred' else 'recoverable'}):")
    c = res["d all"]["coef"]
    s = res["d all"]["coef_cv_std"]
    for f, v in sorted(c.items(), key=lambda kv: -abs(kv[1])):
        flag = "  UNSTABLE" if s and abs(v) < s[f] else ""
        print(f"   {f:22s} {v:+7.3f}  +-{s[f] if s else float('nan'):.3f} across folds{flag}")
    out[kind] = res
    out.setdefault("_scores", {})[kind] = {k: v.tolist() for k, v in test_scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\exp3_detect.json"))
    args = ap.parse_args()
    out: dict = {}
    run_task("pred", args.val, args.test, out)
    run_task("recover", args.val, args.test, out)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
