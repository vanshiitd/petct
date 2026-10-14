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


SPLIT_JSON = Path(__file__).resolve().parent.parent / "splits" / "autopet_v1.json"
_PATIENT: dict[str, str] | None = None


def patient_of(case: str) -> str:
    """Map a scan id to its patient id.

    Cross-validation here groups on patient, not on scan. AutoPET's training
    split holds 738 scans from 630 patients -- 77 patients attend twice or more
    -- and a patient's repeat visits show the same anatomy and usually the same
    lesions, so splitting them across folds lets a fold be scored on a body it
    has already seen. Grouping on `case` (the scan) silently did that for 108
    of the 738 training scans and 6 of the 76 val scans. Test is 200 scans from
    200 patients, so no test number was ever affected.

    A case absent from the split file is an error rather than its own group: the
    fallback would quietly restore the leak for exactly the cases it applies to.
    """
    global _PATIENT
    if _PATIENT is None:
        sp = json.loads(SPLIT_JSON.read_text())
        _PATIENT = {s: p for part in sp["splits"].values()
                    for p, ss in part["cases"].items() for s in ss}
    try:
        return _PATIENT[case]
    except KeyError:
        raise SystemExit(f"{case} is not in {SPLIT_JSON}; cannot group by patient")


def restrict_cases(mode: str) -> set[str] | None:
    """Scans to keep when matching the fitting split to the test population.

    Test is 200 scans from 200 patients, every one a patient with a single
    study in AutoPET. The training split is not: 185 of its 738 scans come from
    77 patients who attend more than once, and a second visit is usually
    post-treatment, so its lesions and its false positives do not look like a
    first presentation. Two restrictions make the comparison fair:

      single  only patients with exactly one study (553 scans), which is the
              population test is actually drawn from
      first   that set plus the earliest visit of each repeat attender (630
              scans), which keeps every patient but only their first scan

    Scan ids carry the study date as their third underscore-separated field,
    which is how 'earliest' is decided.
    """
    if mode == "none":
        return None
    sp = json.loads(SPLIT_JSON.read_text())
    keep = set()
    for part in sp["splits"].values():
        for _, scans in part["cases"].items():
            if len(scans) == 1:
                keep.add(scans[0])
            elif mode == "first":
                keep.add(min(scans, key=lambda s: s.split("_")[2]))
    return keep


def load(path: Path, kind: str, only: set[str] | None = None
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["kind"] == kind and (only is None or r["case"] in only)]
    pos = {"pred": "FP", "recover": "recoverable"}[kind]
    y = np.array([1 if r["status"] == pos else 0 for r in rows])
    g = np.array([patient_of(r["case"]) for r in rows])
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


GBT_GRID = {"max_depth": (2, 3, 4, None), "learning_rate": (0.03, 0.08, 0.15),
            "max_iter": (100, 200, 400), "min_samples_leaf": (20, 50),
            "l2_regularization": (0.0, 1.0)}


def tune_gbt(X, y, groups, seed=0):
    """Pick gradient-boosting hyperparameters by grouped CV on the fitting split.

    Part 1 used a hand-set configuration (depth 3, 150 iterations, lr 0.08)
    chosen while the script fitted on val, so those numbers had seen val. They
    are not reused here. The grid below is searched with the same patient-grouped
    folds used for every other number, on the fitting split alone, and the
    winner is the one config carried to val and test. Early stopping is off so a
    configuration means one thing and the fit is deterministic.

    Returns (best_params, ranked_table).
    """
    from itertools import product
    keys = list(GBT_GRID)
    table = []
    for combo in product(*(GBT_GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        f = lambda p=params: HistGradientBoostingClassifier(
            random_state=seed, early_stopping=False, **p)
        aucs, aps, _ = cv_eval(X, y, groups, f)
        if aucs.size:
            table.append((float(aucs.mean()), float(aucs.std()),
                          float(aps.mean()), params))
    table.sort(key=lambda t: -t[0])
    return table[0][3], table


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


def run_task(kind: str, fit_path: Path, evals: "dict[str, Path]", out: dict,
             fit_only: "set[str] | None" = None) -> None:
    """Fit on one split, score every evaluation split.

    With --fit the detector is fitted on the out-of-fold training cases and
    `val` becomes a held-out check alongside `test`, so val can say whether a
    threshold transfers before test is touched. Without --fit the fitting set
    is val and the only evaluation split is test, which is what Experiment 3
    did originally and reproduces those numbers unchanged.
    """
    rows_v, yv, gv, _ = load(fit_path, kind, fit_only)
    ev = {n: load(p, kind) for n, p in evals.items()}
    names = list(ev)
    head = (f"\n{'='*78}\nTASK '{kind}': fit {len(yv)} candidates "
            f"({yv.sum()} positive, {yv.mean():.1%})")
    for n in names:
        _, y, _, _ = ev[n]
        head += f" | {n} {len(y)} ({y.sum()} positive, {y.mean():.1%})"
    print(head)

    lr = lambda: LogisticRegression(max_iter=5000, C=1.0)
    res, scores = {}, {n: {} for n in names}
    hdr = f"\n{'detector':12s} {'fit CV AUROC':>18s} {'fit CV AP':>16s}"
    for n in names:
        hdr += f" {n + ' AUROC':>13s} {n + ' AP':>11s}"
    print(hdr)
    print("-" * (48 + 26 * len(names)))
    for name, feats in GROUPS.items():
        Xv, med, nm_v = matrix(rows_v, feats, None)
        aucs, aps, coefs = cv_eval(Xv, yv, gv, lr)
        mu, sd = Xv.mean(0), Xv.std(0); sd[sd == 0] = 1.0
        clf = lr(); clf.fit((Xv - mu) / sd, yv)
        line = (f"{name:12s} {aucs.mean():>9.4f} +-{aucs.std():.4f} "
                f"{aps.mean():>9.4f} +-{aps.std():.4f}")
        entry = {"cv_auroc_mean": float(aucs.mean()), "cv_auroc_std": float(aucs.std()),
                 "cv_ap_mean": float(aps.mean()), "n_missing_fit": nm_v,
                 "features": feats,
                 "coef": dict(zip(feats, clf.coef_[0].round(4).tolist())),
                 "coef_cv_std": (dict(zip(feats, coefs.std(0).round(4).tolist()))
                                 if coefs is not None else None)}
        for n in names:
            rows_e, ye, _, _ = ev[n]
            Xe, _, nm_e = matrix(rows_e, feats, med)
            pe = clf.predict_proba((Xe - mu) / sd)[:, 1]
            scores[n][name] = pe
            a, p_ = roc_auc_score(ye, pe), average_precision_score(ye, pe)
            line += f" {a:>13.4f} {p_:>11.4f}"
            entry[f"{n}_auroc"], entry[f"{n}_ap"] = float(a), float(p_)
            entry[f"n_missing_{n}"] = nm_e
        print(line)
        res[name] = entry

    # (e) hand-written rules, thresholds tuned on the fitting split only
    thr = tune_rules(rows_v, yv, gv)
    sv = rule_score(rows_v, thr)
    line = (f"{'e rules':12s} {roc_auc_score(yv, sv):>9.4f} {'(fit)':>9s} "
            f"{average_precision_score(yv, sv):>9.4f}{'':>8s}")
    entry = {"thresholds": thr, "fit_auroc": float(roc_auc_score(yv, sv))}
    for n in names:
        rows_e, ye, _, _ = ev[n]
        se = rule_score(rows_e, thr)
        scores[n]["e rules"] = se
        a, p_ = roc_auc_score(ye, se), average_precision_score(ye, se)
        line += f" {a:>13.4f} {p_:>11.4f}"
        entry[f"{n}_auroc"], entry[f"{n}_ap"] = float(a), float(p_)
    print(line)
    print(f"             thresholds from the fitting split: {thr}")
    res["e rules"] = entry

    # Gradient boosting. Hyperparameters are searched here, by grouped CV on the
    # fitting split only -- never on val or test -- and the search itself is
    # reported so the choice is auditable rather than asserted.
    feats = GROUPS["d all"]
    Xv, med, _ = matrix(rows_v, feats, None)
    best, table = tune_gbt(Xv, yv, gv)
    print(f"\nGBT grid search on the fitting split: {len(table)} configs, "
          f"patient-grouped {len(np.unique(gv))} groups")
    print(f"{'rank':>4s} {'CV AUROC':>18s} {'CV AP':>9s}  params")
    for r, (m, s, ap, prm) in enumerate(table[:5], 1):
        print(f"{r:>4d} {m:>9.4f} +-{s:.4f} {ap:>9.4f}  {prm}")
    worst = table[-1]
    print(f"{'last':>4s} {worst[0]:>9.4f} +-{worst[1]:.4f} {worst[2]:>9.4f}  {worst[3]}")
    print(f"  spread across the grid: {table[0][0] - worst[0]:.4f} AUROC")
    gb = lambda: HistGradientBoostingClassifier(random_state=0,
                                                early_stopping=False, **best)
    aucs, aps, _ = cv_eval(Xv, yv, gv, gb)
    g = gb(); g.fit(Xv, yv)
    line = (f"{'d all (GBT)':12s} {aucs.mean():>9.4f} +-{aucs.std():.4f} "
            f"{aps.mean():>9.4f} +-{aps.std():.4f}")
    entry = {"cv_auroc_mean": float(aucs.mean()), "cv_auroc_std": float(aucs.std()),
             "cv_ap_mean": float(aps.mean()), "params": best,
             "grid_size": len(table),
             "grid_top5": [{"cv_auroc": m, "cv_ap": ap, "params": prm}
                           for m, s, ap, prm in table[:5]],
             "grid_auroc_spread": float(table[0][0] - worst[0])}
    for n in names:
        rows_e, ye, _, _ = ev[n]
        Xe, _, _ = matrix(rows_e, feats, med)
        pg = g.predict_proba(Xe)[:, 1]
        scores[n]["d all (GBT)"] = pg
        a, p_ = roc_auc_score(ye, pg), average_precision_score(ye, pg)
        line += f" {a:>13.4f} {p_:>11.4f}"
        entry[f"{n}_auroc"], entry[f"{n}_ap"] = float(a), float(p_)
    print(line)

    # Permutation importance, measured on the held-out fold of each grouped CV
    # split of the FITTING data. Not on test, and not the training-set impurity
    # gains a tree would otherwise report, which flatter correlated features.
    from sklearn.inspection import permutation_importance
    gkf = GroupKFold(n_splits=5)
    imps = []
    for tr, te in gkf.split(Xv, yv, gv):
        if len(np.unique(yv[tr])) < 2 or len(np.unique(yv[te])) < 2:
            continue
        gg = gb(); gg.fit(Xv[tr], yv[tr])
        r = permutation_importance(gg, Xv[te], yv[te], n_repeats=5,
                                   random_state=0, scoring="roc_auc")
        imps.append(r.importances_mean)
    if imps:
        im = np.array(imps)
        entry["perm_importance"] = dict(zip(feats, im.mean(0).round(5).tolist()))
        entry["perm_importance_cv_std"] = dict(zip(feats, im.std(0).round(5).tolist()))
        print(f"\nGBT permutation importance (AUROC drop, held-out folds of the "
              f"fitting split):")
        order = np.argsort(-im.mean(0))
        for j in order:
            m, s = im.mean(0)[j], im.std(0)[j]
            flag = "  UNSTABLE" if abs(m) < s else ""
            print(f"   {feats[j]:22s} {m:+7.4f}  +-{s:.4f} across folds{flag}")
    res["d all (GBT)"] = entry

    # Three contrasts, because "reasoning beats confidence" has three readings.
    #   d-a  everything over the confidence baseline
    #   d-b  reasoning's contribution on top of ALL uncertainty, which is the
    #        strict form: b already holds confidence and flip stability, so a
    #        gain here cannot be uncertainty wearing a different hat
    #   c-a  reasoning INSTEAD of confidence, which is the replacement claim
    pairs = (("d all", "a conf"), ("d all", "b unc"), ("c reason", "a conf"),
             ("d all (GBT)", "d all"))
    for n in names:
        _, ye, _, _ = ev[n]
        for hi_name, lo_name in pairs:
            d, lo, hi = paired_bootstrap(ye, scores[n][lo_name], scores[n][hi_name])
            print(f"paired bootstrap on {n}, ({hi_name}) - ({lo_name}): {d:+.4f} "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] "
                  f"{'excludes 0' if (lo > 0) == (hi > 0) else 'straddles 0'}")
            key = f"bootstrap_{hi_name}_vs_{lo_name}_{n}".replace(" ", "_")
            res[key] = {"delta_auroc": float(d), "ci_lo": lo, "ci_hi": hi}
        print()

    print(f"\nlogistic coefficients, detector 'd all' (standardised; "
          f"+ means more likely {'FP' if kind == 'pred' else 'recoverable'}):")
    c = res["d all"]["coef"]
    s = res["d all"]["coef_cv_std"]
    for f, v in sorted(c.items(), key=lambda kv: -abs(kv[1])):
        flag = "  UNSTABLE" if s and abs(v) < s[f] else ""
        print(f"   {f:22s} {v:+7.3f}  +-{s[f] if s else float('nan'):.3f} across folds{flag}")
    out[kind] = res
    for n in names:
        out.setdefault(f"_scores_{n}", {})[kind] = {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in scores[n].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--fit", type=Path, default=None,
                    help="candidate CSV to fit on; without it the fitting set is --val")
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\exp3_detect.json"))
    ap.add_argument("--restrict", choices=("none", "single", "first"),
                    default="none",
                    help="restrict the FITTING split to match the test population")
    ap.add_argument("--no-test", action="store_true",
                    help="do not read the test CSV at all")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="feature names to remove from every group (ablation)")
    ap.add_argument("--add", nargs="*", default=[],
                    help="extra feature columns to append to the reason group")
    args = ap.parse_args()
    if args.drop or args.add:
        unknown = [f for f in args.drop if f not in UNC + REASON]
        if unknown:
            raise SystemExit(f"--drop names unknown feature(s): {unknown}")
        for k in GROUPS:
            GROUPS[k] = [f for f in GROUPS[k] if f not in args.drop]
        for k in ("c reason", "d all"):
            GROUPS[k] = GROUPS[k] + [f for f in args.add if f not in GROUPS[k]]
        print(f"ablation: dropped {args.drop or 'nothing'}; added {args.add or 'nothing'}")
        print(f"  'd all' now {len(GROUPS['d all'])} features: {GROUPS['d all']}")
    fit = args.fit or args.val
    evals = {"test": args.test} if args.fit is None else {"val": args.val,
                                                          "test": args.test}
    if args.no_test:
        # the distribution check must not read test at all, not merely refrain
        # from printing it
        evals.pop("test", None)
    only = restrict_cases(args.restrict)
    if only is not None:
        import csv as _csv
        have = {r["case"] for r in _csv.DictReader(
            open(fit, newline="", encoding="utf-8"))}
        kept = have & only
        print(f"restrict={args.restrict}: fitting on {len(kept)} of {len(have)} "
              f"scans in {fit.name}")
    print(f"fitting on {fit.name}; evaluating on "
          + ", ".join(f"{n}={p.name}" for n, p in evals.items()))
    out: dict = {"fit_csv": fit.name,
                 "eval_csv": {n: p.name for n, p in evals.items()}}
    out["restrict"] = args.restrict
    run_task("pred", fit, evals, out, only)
    run_task("recover", fit, evals, out, only)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
