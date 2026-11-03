#!/usr/bin/env python3
"""Phase VII, piece 4: joint diagnosis+selection scorer (addresses reason 6).

The deployed pipeline is two stages: a hard `p_failure >= 0.90` gate decides
*whether* to act, then a selector decides *which* strategy, conditioned on
having passed the gate. `phase5_agent.py` already exposes the mechanism for
collapsing this into one decision: every candidate's predicted value is
compared against `eps_hat` (default 0.0) and the agent only acts if the best
candidate clears it (`c["v"] > self.eps_hat`, `phase5_agent.py` line 218).
The hard gate is a *separate*, independent filter applied before that -- the
joint scorer removes it and lets eps_hat, applied to one calibrated score
that already has `p_failure` as an input feature, do both jobs at once.

This only works if the score is calibrated against zero as "no benefit from
acting" -- which a ranking score (pieces 2-3) is not, but a regression on
`d_qdice` is by construction. So piece 4 goes back to regression (like the
original Phase V action-value model that the audit beat), but:

  - trained on the full p_failure range (training was never gated to begin
    with -- `phase5_action_value.build()` keeps every measured action,
    31,932 of the 34,999 have p_failure < 0.90), so the model has to learn
    where p_failure makes acting worthwhile, rather than being told via a
    fixed threshold;
  - carries piece 3's class-balanced sample weighting forward, so the
    regression is not dominated by `spurious_other`;
  - is deployed with `--min-pfail 0.0` -- no hard gate -- relying entirely on
    the model's own calibrated score and the existing `eps_hat` cutoff.

Same nested-CV discipline as pieces 1-3. Falsified if it does not beat piece
3 on the primary metric at a matched or better safety margin (lesion
sensitivity delta, budget -0.01).
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build
from phase7_shrinkage import FOLDS, primary_metric
from phase7_balanced_ranking import class_balance_weight

GRID = [{"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": 50}
        for d in (3, 5)
        for lr in (0.05, 0.1)
        for it in (200, 400)]


def fit_model(X, y, pred_class_arr, idx: np.ndarray, cfg: dict, seed: int = 0):
    w = class_balance_weight(pred_class_arr, idx)
    m = HistGradientBoostingRegressor(random_state=seed, **cfg)
    m.fit(X[idx], y[idx], sample_weight=w)
    return m


def select_joint(score: np.ndarray, comp_key_arr, mask: np.ndarray, eps: float = 0.0) -> dict:
    """Argmax strategy per component in `mask`, **no p_failure gate** -- the
    component is only included (act) if the argmax score clears `eps`."""
    cands = collections.defaultdict(list)
    for i in np.where(mask)[0]:
        cands[comp_key_arr[i]].append(i)
    chosen = {}
    for key, idxs in cands.items():
        best_i = max(idxs, key=lambda i: score[i])
        if score[best_i] > eps:
            chosen[key] = best_i
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, nargs="+",
                    default=[Path(r"D:\data\reports\strategy_table_oof.csv"),
                             Path(r"D:\data\reports\strategy6_oof.csv")])
    ap.add_argument("--cand", type=Path,
                    default=Path(r"D:\data\reports\cand_cv_oof_pos.csv"))
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--splits", type=Path,
                    default=Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\phase7_joint.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase7_joint.json"))
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    meta, y, X = d["meta"], d["y"], d["X"]
    n = len(meta)
    pred_class_arr = np.array([m["pred_class"] for m in meta], dtype=object)
    comp_key_arr = np.empty(n, dtype=object)
    for i, m in enumerate(meta):
        comp_key_arr[i] = (m["case"], m["component_id"], m["strategy"] == "recover")

    splits = json.loads(args.splits.read_text())
    owner = {}
    for k in FOLDS:
        for s in splits[k]["val"]:
            owner[s] = k
    fold_arr = np.array([owner.get(m["case"], -1) for m in meta])
    print(f"{n} actions, {int((fold_arr < 0).sum())} without a Phase IIIb fold "
          f"(no p_failure filter -- full range trains the act-or-not boundary)")

    outer_report = []
    all_chosen: dict = {}
    for outer in FOLDS:
        te = fold_arr == outer
        tr = ~te & (fold_arr >= 0)
        inner_folds = [f for f in FOLDS if f != outer]
        best_cfg, best_score = None, -np.inf
        for cfg in GRID:
            scores = []
            for held in inner_folds:
                itr = np.where(tr & (fold_arr != held))[0]
                ite = tr & (fold_arr == held)
                if itr.size < 500 or ite.sum() < 50:
                    continue
                m = fit_model(X, y, pred_class_arr, itr, cfg)
                score = m.predict(X)
                chosen = select_joint(score, comp_key_arr, ite)
                if not chosen:
                    continue
                pm = primary_metric(y, chosen)
                scores.append(pm["mean_d_qdice"])
            if scores:
                s = float(np.mean(scores))
                if s > best_score:
                    best_score, best_cfg = s, cfg
        tr_idx = np.where(tr)[0]
        model = fit_model(X, y, pred_class_arr, tr_idx, best_cfg)
        score_all = model.predict(X)
        chosen = select_joint(score_all, comp_key_arr, te)
        pm = primary_metric(y, chosen)
        act_rate = len(chosen) / max(int(te.sum()), 1)
        print(f"  outer fold {outer}: inner-selected {best_cfg}  "
              f"held-out n_acted={pm['n_selected']}  mean d_qdice={pm['mean_d_qdice']:+.5f}  "
              f"precision={pm['precision']:.1f}%  act-rate={100*act_rate:.1f}%")
        outer_report.append({"fold": outer, "cfg": best_cfg, "act_rate": act_rate, **pm})
        all_chosen.update(chosen)

    overall = primary_metric(y, all_chosen)
    print(f"\ncross-fitted overall: n_acted={overall['n_selected']}  "
          f"mean d_qdice={overall['mean_d_qdice']:+.5f}  precision={overall['precision']:.1f}%")
    print("reference (case-level, so far): (b) hand-built +0.0065, "
          "piece 3 (class-balanced ranking) +0.0033 / 59 FP cleared")

    # ---- final production artefact: fit on ALL out-of-fold data -----------
    cfg_counts = collections.Counter(tuple(sorted(o["cfg"].items()))
                                      for o in outer_report)
    final_cfg = dict(cfg_counts.most_common(1)[0][0])
    mask_idx = np.where(fold_arr >= 0)[0]
    final_model = fit_model(X, y, pred_class_arr, mask_idx, final_cfg)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": final_model, "pred_classes": list(d["pred_classes"])},
                    fh)
    args.report.write_text(json.dumps(
        {"outer_folds": outer_report, "overall_crossfit": overall,
         "final_cfg": final_cfg,
         "reference": {"fixed": +0.0033, "hand_built": +0.0065, "learned": +0.0052,
                       "shrinkage_piece1": +0.0029, "ranking_piece2": +0.0033,
                       "balanced_piece3": +0.0033, "balanced_piece3_fp": 59}},
        indent=1, default=float), encoding="utf-8")
    print(f"\nfinal cfg={final_cfg} (mode across outer folds), fit on all out-of-fold data")
    print(f"written {args.out} (avpkl-compatible) and {args.report}")
    print("deploy with --min-pfail 0.0 (no hard gate) -- the model's own score, "
          "compared against eps_hat, is the act-or-not decision")


if __name__ == "__main__":
    main()
