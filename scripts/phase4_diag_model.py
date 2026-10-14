#!/usr/bin/env python3
"""Phase IV step 1b: the diagnosis model P(F_k | O), eq. 19.

Multi-label, observable features only -- no ground truth reaches the inputs.
Two heads, because the two populations are different objects:

  pred      one row per predicted component, labels
            {spurious_physio, spurious_liver_spleen, spurious_other, merging,
             under_seg, over_seg, boundary, fragmentation}
  recover   one row per sub-threshold candidate, label {missed}, i.e. is this
            faint blob actually a lesion the model dropped

Labels come from the ground-truth diagnosis in `phase4_diagnose.py`; the
features come from the candidate CSV and are exactly those Phase IIIb settled
on -- the anatomy-anchored position set, with `z_rel` excluded as a scan-edge
artefact and `u_epi` excluded because it does not exist out-of-fold.

Fitted on out-of-fold with patient-grouped cross-validation. Test is not read
here; it is scored in the final evaluation pass so that the split is touched
once.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, GROUPS, patient_of
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PRED_CLASSES = ["spurious_physio", "spurious_liver_spleen", "spurious_other",
                "merging", "under_seg", "over_seg", "boundary", "fragmentation"]
REC_CLASSES = ["missed"]
DROP = ["z_rel"]
ADD = ["dz_liver_mm", "dz_heart_mm", "dz_bladder_mm", "dz_kidney_mm",
       "edge_body_mm", "edge_scan_mm"]


def labels_from_diag(diag_path: Path) -> tuple[dict, dict]:
    """(case, component_id) -> {class: 0/1}, and the recovery-side truth."""
    recs = pickle.load(open(diag_path, "rb"))
    pred, rec = {}, {}
    for r in recs:
        comp_cls = {c["component_id"]: {k: 0 for k in PRED_CLASSES}
                    for c in r["components"]}
        for c in r["components"]:
            if c["spurious_class"]:
                comp_cls[c["component_id"]][c["spurious_class"]] = 1
            if c["merging"]:
                comp_cls[c["component_id"]]["merging"] = 1
        for l in r["lesions"]:
            for k in ("under_seg", "over_seg", "boundary", "fragmentation"):
                if l[k]:
                    for cid in l["comp_ids"]:
                        if cid in comp_cls:
                            comp_cls[cid][k] = 1
        for cid, d in comp_cls.items():
            pred[(r["case"], cid)] = d
    return pred, rec


def fit_report(X, Y, groups, names, title, Xv=None, Yv=None):
    print(f"\n{title}")
    print(f"  {'class':24s}{'prev':>8s}{'CV AUROC':>11s}{'+-':>8s}{'val':>9s}")
    out = {}
    gkf = GroupKFold(n_splits=5)
    models = {}
    for j, cname in enumerate(names):
        y = Y[:, j]
        prev = float(y.mean())
        if y.sum() < 20 or y.sum() == len(y):
            print(f"  {cname:24s}{prev:8.4f}{'too rare':>11s}")
            out[cname] = {"prevalence": prev, "cv_auroc": None}
            models[cname] = None
            continue
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            m = HistGradientBoostingClassifier(random_state=0, max_depth=3,
                                               learning_rate=0.08, max_iter=200,
                                               early_stopping=False).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        full = HistGradientBoostingClassifier(random_state=0, max_depth=3,
                                              learning_rate=0.08, max_iter=200,
                                              early_stopping=False).fit(X, y)
        models[cname] = full
        va = float("nan")
        if Xv is not None and Yv is not None and Yv[:, j].sum() > 0:
            va = roc_auc_score(Yv[:, j], full.predict_proba(Xv)[:, 1])
        a = np.array(aucs, float)
        print(f"  {cname:24s}{prev:8.4f}{a.mean():11.4f}{a.std():8.4f}{va:9.4f}")
        out[cname] = {"prevalence": prev, "cv_auroc": float(a.mean()),
                      "cv_std": float(a.std()), "val_auroc": float(va)}
    return out, models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-cand", type=Path, required=True)
    ap.add_argument("--fit-diag", type=Path, required=True)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--val-diag", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\phase4_diag_model.json"))
    args = ap.parse_args()

    feats = [f for f in GROUPS["d all"] if f not in DROP]
    feats += [f for f in ADD if f not in feats]
    print(f"{len(feats)} observable features (no ground truth, no u_epi, no z_rel)")

    res, models = {}, {}
    for kind, classes in (("pred", PRED_CLASSES), ("recover", REC_CLASSES)):
        rows, y_status, groups, _ = load(args.fit_cand, kind)
        X, med, _ = matrix(rows, feats, None)
        rows_v, yv_status, _, _ = load(args.val_cand, kind)
        Xv, _, _ = matrix(rows_v, feats, med)
        if kind == "pred":
            lab, _ = labels_from_diag(args.fit_diag)
            labv, _ = labels_from_diag(args.val_diag)
            Y = np.array([[lab.get((r["case"], int(r["component_id"])), {}).get(c, 0)
                           for c in classes] for r in rows], float)
            Yv = np.array([[labv.get((r["case"], int(r["component_id"])), {}).get(c, 0)
                            for c in classes] for r in rows_v], float)
        else:
            # a recovery candidate is "missed" exactly when it is recoverable
            Y = y_status.reshape(-1, 1).astype(float)
            Yv = yv_status.reshape(-1, 1).astype(float)
        print(f"\n{kind}: {X.shape[0]} rows (val {Xv.shape[0]})")
        r, m = fit_report(X, Y, groups, classes,
                          f"P(F_k | O) -- {kind}", Xv, Yv)
        res[kind] = r
        models[kind] = {"models": m, "feats": feats, "median": med}

    args.out.write_text(json.dumps(res, indent=1))
    with open(args.out.with_suffix(".pkl"), "wb") as fh:
        pickle.dump(models, fh)
    print(f"\nwritten to {args.out} and {args.out.with_suffix('.pkl')}")


if __name__ == "__main__":
    main()
