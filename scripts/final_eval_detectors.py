#!/usr/bin/env python3
"""The uncertainty-only detector that section 21's rung 2 is gated by.

The ablation adds one component at a time. Rung 2 is "remove by uncertainty
alone" - Phase III's action A with the confidence/uncertainty detector - and
rung 3 is the same removal gated by the reasoning detector. For those two rows
to differ in exactly one thing, the two detectors must be fitted the same way
and consumed through the same interface, so this writes a pickle with the same
shape as `phase4_diag_model.pkl`:

    {"pred": {"models": {<class>: estimator}, "feats": [...], "median": {...}}}

The only difference is the feature set. Rung 2 uses `exp3_detect.UNC` - p_mean,
p_max, uent_mean, uent_max, ustab_mean, ustab_max - and nothing else: no SUV, no
anatomy, no shape. One head named `spurious_other`, so the agent's "take the
strongest predicted failure" logic works unchanged and the gate means the same
thing (P(this component is a false positive) >= 0.90).

Hyperparameters are Phase IIIb's, not re-tuned: HistGradientBoosting, depth 3,
lr 0.08, 200 iterations, no early stopping. Fitted on out-of-fold only, with
patient-grouped CV reported for the record.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, UNC, patient_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-cand", type=Path, required=True)
    ap.add_argument("--val-cand", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\final_unc_detector.pkl"))
    args = ap.parse_args()

    rows, y, g, _ = load(args.fit_cand, "pred")
    X, med, n_missing = matrix(rows, UNC, None)
    print(f"{len(rows)} out-of-fold components, {int(y.sum())} false positives "
          f"({y.mean():.3f}), {len(UNC)} uncertainty features, "
          f"{n_missing} imputed cells")

    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(random_state=0, max_depth=3,
                                           learning_rate=0.08, max_iter=200,
                                           early_stopping=False).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    a = np.array(aucs, float)
    print(f"patient-grouped CV AUROC {a.mean():.4f} +- {a.std():.4f}")

    full = HistGradientBoostingClassifier(random_state=0, max_depth=3,
                                          learning_rate=0.08, max_iter=200,
                                          early_stopping=False).fit(X, y)
    va = float("nan")
    if args.val_cand:
        vr, vy, _, _ = load(args.val_cand, "pred")
        Xv, _, _ = matrix(vr, UNC, med)
        va = float(roc_auc_score(vy, full.predict_proba(Xv)[:, 1]))
        print(f"validation AUROC {va:.4f}")

    # Same container shape as phase4_diag_model.pkl so the agent needs no
    # special case. The single head is named `spurious_other` because that is
    # what removal acts on, and the agent's policy lookup keys on the name.
    obj = {"pred": {"models": {"spurious_other": full}, "feats": list(UNC),
                    "median": med},
           "recover": {"models": {"missed": None}, "feats": list(UNC),
                       "median": med}}
    with open(args.out, "wb") as fh:
        pickle.dump(obj, fh)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"features": list(UNC), "cv_auroc": float(a.mean()),
         "cv_sd": float(a.std()), "val_auroc": va, "n": len(rows)}, indent=1),
        encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
