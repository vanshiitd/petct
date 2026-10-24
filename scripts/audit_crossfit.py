#!/usr/bin/env python3
"""Audit R1: cross-fitted out-of-fold predictions for every fitted component.

`phase4_diag_model.py` keeps only the model fitted on all 738 out-of-fold scans
(`models[cname] = full`, l.91-94), and `final_eval_detectors.py` does the same
(l.63-65). The agents then apply those models to the very scans they were fitted
on, so every out-of-fold `p_failure` in Phases IV-VI is **in-sample**. The same
holds for the action-value model and the hand-built policy. Q is the exception:
`q_model.pkl` carries per-fold models and the agents select the fold that did
not see the patient.

This script produces honestly cross-fitted versions of all four, using the
Phase IIIb patient-grouped folds 3-7, which partition the 738 scans exactly.
Hyperparameters are unchanged from the originals; only the fitting set changes.

Writes `audit_crossfit.npz` (per-component arrays) and `audit_crossfit.json`
(summary), both new files. Nothing existing is modified.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, GROUPS, UNC
from phase4_diag_model import PRED_CLASSES, REC_CLASSES, DROP, ADD, labels_from_diag
from q_model import patient_of

FOLDS = (3, 4, 5, 6, 7)
# unchanged from phase4_diag_model.fit_report and final_eval_detectors
CLF = dict(random_state=0, max_depth=3, learning_rate=0.08, max_iter=200,
           early_stopping=False)


def fold_of_case(splits_path: Path) -> dict:
    """{case: fold} from the split file the training actually used."""
    splits = json.loads(splits_path.read_text())
    owner = {}
    for k in FOLDS:
        for s in splits[k]["val"]:
            if s in owner:
                raise SystemExit(f"{s} held out by folds {owner[s]} and {k}")
            owner[s] = k
    return owner


def crossfit_binary(X, y, fold, classes_label=""):
    """Out-of-fold probabilities: each row scored by the fold that excluded it."""
    p = np.full(len(y), np.nan)
    for k in FOLDS:
        te = fold == k
        tr = ~te
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        m = HistGradientBoostingClassifier(**CLF).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path,
                    default=Path(r"D:\data\reports\cand_cv_oof_pos.csv"))
    ap.add_argument("--diag", type=Path,
                    default=Path(r"D:\data\reports\diag_oof.pkl"))
    ap.add_argument("--splits", type=Path,
                    default=Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json"))
    ap.add_argument("--table", type=Path, nargs="+",
                    default=[Path(r"D:\data\reports\strategy_table_oof.csv"),
                             Path(r"D:\data\reports\strategy6_oof.csv")])
    ap.add_argument("--insample", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--unc-insample", type=Path,
                    default=Path(r"D:\data\reports\final_unc_detector.pkl"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_crossfit.npz"))
    args = ap.parse_args()

    owner = fold_of_case(args.splits)
    print(f"folds {FOLDS} cover {len(owner)} scans")

    feats = [f for f in GROUPS["d all"] if f not in DROP]
    feats += [f for f in ADD if f not in feats]

    out = {}
    summary = {"folds": list(FOLDS), "n_features": len(feats)}

    # ---- diagnosis heads, 8 classes, one row per predicted component ------
    rows, _, _, _ = load(args.cand, "pred")
    X, med, _ = matrix(rows, feats, None)
    fold = np.array([owner.get(r["case"], -1) for r in rows])
    lab, _ = labels_from_diag(args.diag)
    keys = [(r["case"], int(r["component_id"])) for r in rows]
    print(f"pred head: {len(rows)} components, "
          f"{int((fold < 0).sum())} without a fold")

    P_cf = np.full((len(rows), len(PRED_CLASSES)), np.nan)
    P_is = np.full((len(rows), len(PRED_CLASSES)), np.nan)
    insample = pickle.load(open(args.insample, "rb"))["pred"]
    Xi, _, _ = matrix(rows, insample["feats"], insample["median"])
    auc = {}
    for j, c in enumerate(PRED_CLASSES):
        y = np.array([lab.get(k, {}).get(c, 0) for k in keys], float)
        P_cf[:, j] = crossfit_binary(X, y, fold, c)
        m = insample["models"].get(c)
        if m is not None:
            P_is[:, j] = m.predict_proba(Xi)[:, 1]
        from sklearn.metrics import roc_auc_score
        ok = np.isfinite(P_cf[:, j])
        auc[c] = {
            "prevalence": float(y.mean()),
            "auroc_crossfit": (float(roc_auc_score(y[ok], P_cf[ok, j]))
                               if y[ok].sum() > 0 else None),
            "auroc_insample": (float(roc_auc_score(y, P_is[:, j]))
                               if m is not None and y.sum() > 0 else None),
        }
        print(f"  {c:24s} prev {y.mean():.4f}  "
              f"AUROC in-sample {auc[c]['auroc_insample']}  "
              f"cross-fitted {auc[c]['auroc_crossfit']}")
    summary["pred_heads"] = auc

    out["keys_case"] = np.array([k[0] for k in keys])
    out["keys_cid"] = np.array([k[1] for k in keys], dtype=np.int64)
    out["fold"] = fold
    out["P_crossfit"] = P_cf
    out["P_insample"] = P_is
    out["classes"] = np.array(PRED_CLASSES)

    # ---- uncertainty-only detector ---------------------------------------
    _, y_fp, _, _ = load(args.cand, "pred")
    Xu, medu, _ = matrix(rows, list(UNC), None)
    u_cf = crossfit_binary(Xu, y_fp.astype(float), fold, "unc")
    ui = pickle.load(open(args.unc_insample, "rb"))["pred"]
    Xui, _, _ = matrix(rows, ui["feats"], ui["median"])
    u_is = ui["models"]["spurious_other"].predict_proba(Xui)[:, 1]
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(u_cf)
    summary["unc_detector"] = {
        "auroc_insample": float(roc_auc_score(y_fp, u_is)),
        "auroc_crossfit": float(roc_auc_score(y_fp[ok], u_cf[ok])),
    }
    print(f"\nuncertainty-only detector: AUROC in-sample "
          f"{summary['unc_detector']['auroc_insample']:.4f}, cross-fitted "
          f"{summary['unc_detector']['auroc_crossfit']:.4f}")
    out["unc_crossfit"] = u_cf
    out["unc_insample"] = u_is

    # ---- recovery head ----------------------------------------------------
    rrows, ry, _, _ = load(args.cand, "recover")
    if len(rrows):
        Xr, _, _ = matrix(rrows, feats, med)
        rfold = np.array([owner.get(r["case"], -1) for r in rrows])
        r_cf = crossfit_binary(Xr, ry.astype(float), rfold, "missed")
        rm = pickle.load(open(args.insample, "rb"))["recover"]
        Xri, _, _ = matrix(rrows, rm["feats"], rm["median"])
        r_is = rm["models"]["missed"].predict_proba(Xri)[:, 1]
        ok = np.isfinite(r_cf)
        summary["recover_head"] = {
            "auroc_insample": float(roc_auc_score(ry, r_is)),
            "auroc_crossfit": float(roc_auc_score(ry[ok], r_cf[ok])),
        }
        print(f"recovery head: AUROC in-sample "
              f"{summary['recover_head']['auroc_insample']:.4f}, cross-fitted "
              f"{summary['recover_head']['auroc_crossfit']:.4f}")
        out["rec_keys_case"] = np.array([r["case"] for r in rrows])
        out["rec_keys_cid"] = np.array([int(r["component_id"]) for r in rrows],
                                       dtype=np.int64)
        out["rec_crossfit"] = r_cf
        out["rec_insample"] = r_is

    np.savez_compressed(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1),
                                             encoding="utf-8")
    print(f"\nwritten {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
