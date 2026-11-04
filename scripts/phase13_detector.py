#!/usr/bin/env python3
"""Phase XIII: refit the per-component detector with context features.

See `reports/phase13_context_detector_plan.md`. Same eight class heads, same
labels and same `CLF`; the only change is the feature set (the current one
plus `phase13_context_features.NEW`). Nothing is tuned.

Route checks before anything new is written:
  * out-of-fold, the old features reproduce `audit_crossfit.npz` exactly;
  * on validation, old features fitted on all out-of-fold rows reproduce the
    deployed detector (`phase4_diag_model.pkl`).

Writes, all new files:
  phase13_det_{oof,val,test}.npz   --pfail-table for the hand-built agent
  phase13_det.json                 detection descriptives
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase4_diag_model import PRED_CLASSES, labels_from_diag
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, fit_stage1,
                                     predict_stage1, stage1_crossfit, calibration,
                                     write_table)
from phase13_context_features import NEW

GROUPS = {"contrast": ["shell_suv_mean", "tbr_mean", "tbr_max", "edge_contrast"],
          "texture": ["suv_cv", "frac_hot50", "hu_std", "frac_bone_hu", "frac_air_hu"],
          "neighbourhood": ["shell_frac_air", "shell_frac_bone", "shell_frac_physio",
                            "shell_frac_liver"],
          "model_profile": ["p_frac90", "p_shell_mean", "uent_shell_mean"],
          "symmetry": ["mirror_suv_ratio"],
          "within_case": ["suvmax_rank_case", "suvmax_over_case_p95", "n_pred_case",
                          "dist_nearest_pred_mm"]}
assert sorted(sum(GROUPS.values(), [])) == sorted(NEW)


def labels(rows, diag):
    lab, _ = labels_from_diag(diag)
    keys = [(r["case"], int(r["component_id"])) for r in rows]
    return keys, np.array([[lab.get(k, {}).get(c, 0) for c in PRED_CLASSES]
                           for k in keys], float)


def detection(P, Y):
    out = {}
    for j, c in enumerate(PRED_CLASSES):
        y = Y[:, j]
        out[c] = ({"auroc": float(roc_auc_score(y, P[:, j])),
                   "ap": float(average_precision_score(y, P[:, j])),
                   "n_pos": int(y.sum())} if 0 < y.sum() < len(y) else None)
    pm = P.max(1)
    hit = Y[np.arange(len(Y)), P.argmax(1)]
    out["max_class"] = calibration(pm, hit)
    out["max_class"].pop("reliability")
    out["n_at_gate"] = int((pm >= BASE_GATE).sum())
    return out


def main() -> None:
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    old = features()
    new = old + NEW
    rows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    keys, Y = labels(rows, R / "diag_oof.pkl")
    fold = np.array([owner.get(r["case"], -1) for r in rows])
    assert (fold >= 0).all()
    X0, med0, _ = matrix(rows, old, None)
    X1, med1, _ = matrix(rows, new, None)

    # ---- route check 1: old features reproduce audit_crossfit -----------
    z = np.load(R / "audit_crossfit.npz", allow_pickle=True)
    ref = {(str(c), int(i)): p for c, i, p in
           zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
    P0 = stage1_crossfit(X0, Y, fold, list(FOLDS))
    dev = float(np.abs(P0 - np.array([ref[k] for k in keys])).max())
    print(f"route check, out-of-fold: max |P_old - audit_crossfit| = {dev:.2e}")
    assert dev < 1e-9, "old-feature cross-fit does not reproduce audit_crossfit"
    rec_keys = [(str(c), int(i)) for c, i in zip(z["rec_keys_case"], z["rec_keys_cid"])]
    rec_cf = np.asarray(z["rec_crossfit"], float)

    # ---- new detector, out-of-fold --------------------------------------
    P1 = stage1_crossfit(X1, Y, fold, list(FOLDS))
    report = {"plan": "reports/phase13_context_detector_plan.md", "new_features": NEW,
              "route_dev_oof": dev,
              "oof": {"old": detection(P0, Y), "new": detection(P1, Y)}}
    write_table(R / "phase13_det_oof.npz", keys, P1, np.ones(len(keys), bool),
                rec_keys, rec_cf)

    # permutation importance of the new groups on the spurious_other head
    j = PRED_CLASSES.index("spurious_other")
    rng = np.random.default_rng(2026)
    imp = {g: [] for g in GROUPS}
    for k in FOLDS:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        m = fit_stage1(X1, Y, tr)[j]
        base = roc_auc_score(Y[te, j], m.predict_proba(X1[te])[:, 1])
        for g, cols in GROUPS.items():
            Xp = X1[te].copy()
            for c in cols:
                Xp[:, new.index(c)] = rng.permutation(Xp[:, new.index(c)])
            imp[g].append(base - roc_auc_score(Y[te, j], m.predict_proba(Xp)[:, 1]))
    report["perm_importance_spurious_other_auroc_drop"] = {
        g: float(np.mean(v)) for g, v in imp.items()}

    # ---- production heads (validation, test) ----------------------------
    all_idx = np.arange(len(rows))
    M0, M1 = fit_stage1(X0, Y, all_idx), fit_stage1(X1, Y, all_idx)
    D = pickle.load(open(R / "phase4_diag_model.pkl", "rb"))["pred"]
    for split in ("val", "test"):
        rs, _, _, _ = load(R / f"cand_cv_{split}_ctx.csv", "pred")
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        Xs0, _, _ = matrix(rs, old, med0)
        Xs1, _, _ = matrix(rs, new, med1)
        Ps0, Ps1 = predict_stage1(M0, Xs0), predict_stage1(M1, Xs1)
        hb = np.load(R / f"phase11_eag_hb_{split}.npz", allow_pickle=True)
        if split == "val":
            Xd, _, _ = matrix(rs, D["feats"], D["median"])
            Pd = np.column_stack([D["models"][c].predict_proba(Xd)[:, 1]
                                  for c in PRED_CLASSES])
            dv = float(np.abs(Ps0 - Pd).max())
            print(f"route check, validation: max |P_old - deployed detector| = {dv:.2e}")
            assert dv < 1e-9, "old-feature refit does not reproduce the deployed detector"
            report["route_dev_val"] = dv
            _, Yv = labels(rs, R / "diag_val.pkl")
            report["val"] = {"old": detection(Ps0, Yv), "new": detection(Ps1, Yv)}
        write_table(R / f"phase13_det_{split}.npz", ks, Ps1, np.ones(len(ks), bool),
                    [(str(c), int(i)) for c, i in zip(hb["rec_keys_case"], hb["rec_keys_cid"])],
                    np.asarray(hb["rec_crossfit"], float))
        report[f"{split}_n_at_gate"] = {"old": int((Ps0.max(1) >= BASE_GATE).sum()),
                                        "new": int((Ps1.max(1) >= BASE_GATE).sum())}

    for part in ("oof", "val"):
        print(f"\n{part}: AUROC old -> new")
        for c in PRED_CLASSES:
            o, n = report[part]["old"][c], report[part]["new"][c]
            if o and n:
                print(f"  {c:22s} {o['auroc']:.3f} -> {n['auroc']:.3f}   "
                      f"AP {o['ap']:.3f} -> {n['ap']:.3f}   (n+ {o['n_pos']})")
        print(f"  at 0.90 gate: {report[part]['old']['n_at_gate']} -> "
              f"{report[part]['new']['n_at_gate']}")
    print(f"\nspurious_other permutation importance: "
          f"{report['perm_importance_spurious_other_auroc_drop']}")
    (R / "phase13_det.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print(f"written {R / 'phase13_det.json'}")


if __name__ == "__main__":
    main()
