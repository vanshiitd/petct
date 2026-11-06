#!/usr/bin/env python3
"""Phase XVIII: the two frozen PSMA tables, hand-built and Phase XVII.

See `reports/phase18_psma_plan.md`. Nothing is fitted or selected on PSMA:
  * hand-built -- the stage-1 detector of `phase4_diag_model.pkl` gated at
    0.90, with its recovery head, exactly as `phase11_eag_hb_{val,test}`;
  * Phase XVII -- the production update of `phase17_ltt.py` (configuration
    read from `phase17_ltt.json`; detectors, calibrators and fence refitted on
    all FDG out-of-fold data exactly as there), over the hand-built actions.
Route check: the refitted old detector must reproduce the pickled one exactly.

Writes phase18_psma_{hb,ltt}.npz and phase18_psma_tables.json.
"""
from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, fit_stage1,
                                     predict_stage1, write_table)
from phase4_diag_model import PRED_CLASSES
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import cal_fit, cal_apply
from phase15_ood import case_fence
from phase16_combined import union

CAND = R / "cand_psma_pos.csv"
CTX = R / "cand_psma_ctx.csv"


def main() -> None:
    cfg = tuple(json.loads((R / "phase17_ltt.json").read_text())["config"])
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    old, new = features(), features() + NEW
    rows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    _, Y = labels(rows, R / "diag_oof.pkl")
    X0, med0, _ = matrix(rows, old, None)
    X1, med1, _ = matrix(rows, new, None)
    fold = np.array([owner[r["case"]] for r in rows])
    from phase15_ood import cal_crossfit_ind  # noqa: F401  (documents the route)
    ncase = collections.Counter(r["case"] for r in rows)
    fz = case_fence(rows, list(range(len(rows))))
    ind = np.array([ncase[r["case"]] for r in rows]) <= fz
    from phase11_experience_gate import stage1_crossfit
    P0 = stage1_crossfit(X0, Y, fold, list(FOLDS))
    P1 = stage1_crossfit(X1, Y, fold, list(FOLDS))
    C0, C1 = cal_fit(P0[ind], Y[ind]), cal_fit(P1[ind], Y[ind])
    M0, M1 = fit_stage1(X0, Y, np.arange(len(rows))), fit_stage1(X1, Y, np.arange(len(rows)))

    D = pickle.load(open(R / "phase4_diag_model.pkl", "rb"))
    Dp, Dr = D["pred"], D["recover"]
    rs, _, _, _ = load(CTX, "pred")
    ks = [(r["case"], int(r["component_id"])) for r in rs]
    Xd, _, _ = matrix(rs, Dp["feats"], Dp["median"])
    Ps = np.column_stack([Dp["models"][c].predict_proba(Xd)[:, 1] for c in PRED_CLASSES])
    Ps0 = predict_stage1(M0, matrix(rs, old, med0)[0])
    Ps1 = predict_stage1(M1, matrix(rs, new, med1)[0])
    dev = float(np.abs(Ps - Ps0).max()) if len(rs) else 0.0
    print(f"route check: max |P_pickled - P_refit| = {dev:.2e}")
    assert dev < 1e-9

    rr, _, _, _ = load(CAND, "recover")
    rk = [(r["case"], int(r["component_id"])) for r in rr]
    Xr, _, _ = matrix(rr, Dr["feats"], Dr["median"])
    prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)

    nc = collections.Counter(r["case"] for r in rs)
    nr = np.array([nc[r["case"]] for r in rs])
    P, act, added = union(cfg, Ps0, cal_apply(C0, Ps0), cal_apply(C1, Ps1), nr > 1, nr <= fz)
    hb_act = Ps.max(1) >= BASE_GATE
    assert np.array_equal(act & ~added, hb_act), "hand-built actions must be kept"
    write_table(R / "phase18_psma_hb.npz", ks, Ps, hb_act, rk, prs)
    write_table(R / "phase18_psma_ltt.npz", ks, P, act, rk, prs)
    rep = {"config": cfg, "fence": fz, "n_components": len(rs), "n_cases": len(nc),
           "handbuilt_acted": int(hb_act.sum()), "added": int(added.sum()),
           "abstain_cases": sorted({r["case"] for r, n in zip(rs, nr) if n > fz}),
           "recovery_candidates": int((prs >= BASE_GATE).sum())}
    (R / "phase18_psma_tables.json").write_text(json.dumps(rep, indent=1, default=float))
    print({k: v for k, v in rep.items() if k != "abstain_cases"},
          f"abstaining cases {len(rep['abstain_cases'])}")


if __name__ == "__main__":
    main()
