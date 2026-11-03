#!/usr/bin/env python3
"""Phase VII, piece 3: rebalanced / stratified training (addresses reason 3).

Piece 2 fixed the signal-to-noise problem (cross-fitted Spearman 0.31 -> 0.40)
but still collapsed onto `topology_remove` for 606/607 gated components --
confirmed directly in the experience log, not inferred (see
`reports/phase7_piece2_result.md`). The cause is visible in the training
join itself:

    spurious_other          19957
    boundary                 7475
    fragmentation             3393
    under_seg                 1575
    missed                    1086
    merging                    563
    over_seg                   530
    spurious_liver_spleen      272
    spurious_physio            148

`spurious_other` outnumbers `spurious_physio` 135 to 1. A ranking loss that
aggregates gradient equally over every (case, component) group bets on
whatever action is correct for the largest class almost everywhere, including
on the minority classes where a different action wins by a wide margin.

This re-fits piece 2's exact model (same features, same lambdarank objective,
same within-component relevance labels, same nested-CV discipline) with one
change: a per-row sample weight, `1 / count(pred_class)` within the *training*
rows only (never computed on the held-out fold, validation, or test),
normalised to mean 1 so the loss function sees every failure class as equally
important regardless of how often it occurs. This is the weighting-based
equivalent of a stratified resample for a gradient-boosted ranker -- it
reaches the same effect (equal per-class influence on the fitted trees)
without duplicating rows, which would distort the tree-building statistics
LightGBM uses (histogram bin counts, leaf sizes) in ways a plain weight does
not.

Falsified if it does not reduce the selection's concentration on a single
strategy (measured directly from the accepted-action distribution) without
costing the primary metric.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build
from phase7_shrinkage import primary_metric
from phase7_ranking import GRID, FOLDS, GATE, group_blocks, relevance_labels, \
    select_per_component


def class_balance_weight(pred_class_arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """1 / count(class) over `idx` only, normalised to mean 1 over `idx`."""
    counts = collections.Counter(pred_class_arr[i] for i in idx)
    w = np.array([1.0 / counts[pred_class_arr[i]] for i in idx])
    return w * (len(w) / w.sum())


def fit_ranker(X, y, comp_key_arr, pred_class_arr, idx: np.ndarray, cfg: dict,
               seed: int = 0):
    order, sizes = group_blocks(idx, comp_key_arr)
    labels = relevance_labels(order, sizes, y)
    weights = class_balance_weight(pred_class_arr, order)
    m = LGBMRanker(objective="lambdarank", random_state=seed, verbosity=-1, **cfg)
    m.fit(X[order], labels, group=sizes, sample_weight=weights)
    return m


def strategy_concentration(chosen: dict, meta: list) -> dict:
    """Share of accepted (selected) actions taken by the single most common
    strategy -- the direct, falsifiable measure of the piece 2 collapse."""
    c = collections.Counter(meta[i]["strategy"] for i in chosen.values())
    total = sum(c.values())
    top_strat, top_n = c.most_common(1)[0] if c else (None, 0)
    return {"n": total, "top_strategy": top_strat, "top_n": top_n,
            "top_share": top_n / total if total else 0.0,
            "distribution": dict(c)}


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
                    default=Path(r"D:\data\reports\phase7_balanced_ranking.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase7_balanced_ranking.json"))
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    meta, y, X = d["meta"], d["y"], d["X"]
    n = len(meta)
    pfail_arr = np.array([m["p_failure"] for m in meta], dtype=float)
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
    print(f"{n} actions, {int((fold_arr < 0).sum())} without a Phase IIIb fold")

    outer_report = []
    all_chosen: dict = {}
    pooled_pred = np.full(n, np.nan)
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
                m = fit_ranker(X, y, comp_key_arr, pred_class_arr, itr, cfg)
                score = m.predict(X)
                chosen = select_per_component(score, comp_key_arr, pfail_arr, ite)
                if not chosen:
                    continue
                pm = primary_metric(y, chosen)
                scores.append(pm["mean_d_qdice"])
            if scores:
                s = float(np.mean(scores))
                if s > best_score:
                    best_score, best_cfg = s, cfg
        tr_idx = np.where(tr)[0]
        model = fit_ranker(X, y, comp_key_arr, pred_class_arr, tr_idx, best_cfg)
        score_all = model.predict(X)
        pooled_pred[te] = score_all[te]
        chosen = select_per_component(score_all, comp_key_arr, pfail_arr, te)
        pm = primary_metric(y, chosen)
        rho = float(spearmanr(score_all[te], y[te]).statistic)
        conc = strategy_concentration(chosen, meta)
        print(f"  outer fold {outer}: inner-selected {best_cfg}  "
              f"held-out n={pm['n_selected']}  mean d_qdice={pm['mean_d_qdice']:+.5f}  "
              f"precision={pm['precision']:.1f}%  Spearman={rho:.4f}  "
              f"top strategy {conc['top_strategy']} ({100*conc['top_share']:.1f}%)")
        outer_report.append({"fold": outer, "cfg": best_cfg, "spearman": rho,
                              "concentration": conc, **pm})
        all_chosen.update(chosen)

    overall = primary_metric(y, all_chosen)
    overall_rho = float(spearmanr(pooled_pred[fold_arr >= 0],
                                   y[fold_arr >= 0]).statistic)
    overall_conc = strategy_concentration(all_chosen, meta)
    print(f"\ncross-fitted overall: n={overall['n_selected']}  "
          f"mean d_qdice={overall['mean_d_qdice']:+.5f}  "
          f"precision={overall['precision']:.1f}%  pooled Spearman={overall_rho:.4f}")
    print(f"strategy concentration: top={overall_conc['top_strategy']} "
          f"{100*overall_conc['top_share']:.1f}%  (piece 2 reference: topology_remove 99.8%)")
    print(f"distribution: {overall_conc['distribution']}")

    # ---- final production artefact: fit on ALL out-of-fold data -----------
    cfg_counts = collections.Counter(tuple(sorted(o["cfg"].items()))
                                      for o in outer_report)
    final_cfg = dict(cfg_counts.most_common(1)[0][0])
    mask_idx = np.where(fold_arr >= 0)[0]
    final_model = fit_ranker(X, y, comp_key_arr, pred_class_arr, mask_idx, final_cfg)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": final_model, "pred_classes": list(d["pred_classes"])},
                    fh)
    args.report.write_text(json.dumps(
        {"outer_folds": outer_report, "overall_crossfit": overall,
         "overall_spearman": overall_rho, "overall_concentration": overall_conc,
         "final_cfg": final_cfg,
         "reference": {"fixed": +0.0033, "hand_built": +0.0065, "learned": +0.0052,
                       "shrinkage_piece1": +0.0029, "ranking_piece2": +0.0033,
                       "ranking_piece2_concentration": 0.998}},
        indent=1, default=float), encoding="utf-8")
    print(f"\nfinal cfg={final_cfg} (mode across outer folds), fit on all out-of-fold data")
    print(f"written {args.out} (avpkl-compatible) and {args.report}")


if __name__ == "__main__":
    main()
