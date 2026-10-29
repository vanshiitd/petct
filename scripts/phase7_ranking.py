#!/usr/bin/env python3
"""Phase VII, piece 2: pairwise ranking + case-level credit assignment.

Reasons 1-2 from the retrospective: the action-value regressor was fit on
`d_qdice` as a raw regression target whose median effect is ~0.0002 -- far
below the label noise floor -- and its own cross-fitted ranking quality
(Spearman ~0.31) was weak as a result. But the agent never needs the
*magnitude* of an action's effect: `phase5_agent.py` always takes the argmax
strategy among the candidates measured for one component. What has to be
right is the *order* of strategies within a component, not the regression
target.

This replaces the regression with LightGBM's `lambdarank` objective, grouped
by `(case, component_id)` -- every row carrying the same group id is a
candidate strategy measured for the same component, i.e. credit is assigned
at the level the agent actually decides at, not pooled across the whole
dataset the way a plain regression loss does.

Relevance labels are the within-component dense rank of `d_qdice` (ties get
the same label, so lambdarank spends no gradient ordering outcomes that are
in fact equal) -- never the raw delta, which is exactly the signal/noise
problem this piece exists to fix.

Same cross-fitted/nested-CV discipline as piece 1 (`phase7_shrinkage.py`):
outer = Phase IIIb folds 3,4,5,6,7; inner = leave-one-(training)-fold-out,
hyperparameters selected on out-of-fold data alone. The fitted artefact is a
`.predict(X)`-compatible object (LGBMRanker itself, unmodified), so it drops
into `phase5_agent.py --avpkl` with zero agent changes, same 45-column `X` as
`phase5_action_value.build()`.
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
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build, STRATEGIES
from phase7_shrinkage import primary_metric

FOLDS = (3, 4, 5, 6, 7)
GATE = 0.90  # matches the deployed p_failure gate throughout

GRID = [{"n_estimators": e, "num_leaves": nl, "learning_rate": lr,
         "min_child_samples": mc}
        for e in (100, 300)
        for nl in (15, 31)
        for lr in (0.05, 0.1)
        for mc in (20,)]


def group_blocks(idx: np.ndarray, comp_key_arr: np.ndarray):
    """Sort `idx` by component key; return (sorted_idx, group_sizes)."""
    order = idx[np.argsort([str(comp_key_arr[i]) for i in idx], kind="stable")]
    keys = [comp_key_arr[i] for i in order]
    sizes, cnt, cur = [], 0, object()
    for k in keys:
        if k != cur:
            if cnt:
                sizes.append(cnt)
            cur, cnt = k, 1
        else:
            cnt += 1
    if cnt:
        sizes.append(cnt)
    return order, np.array(sizes)


def relevance_labels(order: np.ndarray, sizes: np.ndarray, y: np.ndarray):
    """Within-component dense rank of y, ties collapsed to the same label."""
    labels = np.zeros(len(order), dtype=int)
    pos = 0
    for sz in sizes:
        block = order[pos:pos + sz]
        v = y[block]
        finite = np.isfinite(v)
        v2 = np.where(finite, v, np.nanmin(v[finite]) if finite.any() else 0.0)
        labels[pos:pos + sz] = rankdata(v2, method="dense").astype(int) - 1
        pos += sz
    return labels


def fit_ranker(X, y, comp_key_arr, idx: np.ndarray, cfg: dict, seed: int = 0):
    order, sizes = group_blocks(idx, comp_key_arr)
    labels = relevance_labels(order, sizes, y)
    m = LGBMRanker(objective="lambdarank", random_state=seed, verbosity=-1, **cfg)
    m.fit(X[order], labels, group=sizes)
    return m


def select_per_component(score: np.ndarray, comp_key_arr, pfail_arr,
                          mask: np.ndarray, gate: float = GATE) -> dict:
    """Argmax strategy per gated component within `mask`, by global index."""
    cands = collections.defaultdict(list)
    for i in np.where(mask & (pfail_arr >= gate))[0]:
        cands[comp_key_arr[i]].append(i)
    return {key: max(idxs, key=lambda i: score[i]) for key, idxs in cands.items()}


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
                    default=Path(r"D:\data\reports\phase7_ranking.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase7_ranking.json"))
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    meta, y, X = d["meta"], d["y"], d["X"]
    n = len(meta)
    pfail_arr = np.array([m["p_failure"] for m in meta], dtype=float)
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
                m = fit_ranker(X, y, comp_key_arr, itr, cfg)
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
        model = fit_ranker(X, y, comp_key_arr, tr_idx, best_cfg)
        score_all = model.predict(X)
        pooled_pred[te] = score_all[te]
        chosen = select_per_component(score_all, comp_key_arr, pfail_arr, te)
        pm = primary_metric(y, chosen)
        rho = float(spearmanr(score_all[te], y[te]).statistic)
        print(f"  outer fold {outer}: inner-selected {best_cfg}  "
              f"held-out n={pm['n_selected']}  mean d_qdice={pm['mean_d_qdice']:+.5f}  "
              f"precision={pm['precision']:.1f}%  Spearman={rho:.4f}")
        outer_report.append({"fold": outer, "cfg": best_cfg, "spearman": rho, **pm})
        all_chosen.update(chosen)

    overall = primary_metric(y, all_chosen)
    overall_rho = float(spearmanr(pooled_pred[fold_arr >= 0],
                                   y[fold_arr >= 0]).statistic)
    print(f"\ncross-fitted overall: n={overall['n_selected']}  "
          f"mean d_qdice={overall['mean_d_qdice']:+.5f}  "
          f"precision={overall['precision']:.1f}%  pooled Spearman={overall_rho:.4f}")
    print("reference: existing learned action-value model cross-fitted Spearman ~0.31")

    # ---- final production artefact: fit on ALL out-of-fold data -----------
    cfg_counts = collections.Counter(tuple(sorted(o["cfg"].items()))
                                      for o in outer_report)
    final_cfg = dict(cfg_counts.most_common(1)[0][0])
    mask_idx = np.where(fold_arr >= 0)[0]
    final_model = fit_ranker(X, y, comp_key_arr, mask_idx, final_cfg)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": final_model, "pred_classes": list(d["pred_classes"])},
                    fh)
    args.report.write_text(json.dumps(
        {"outer_folds": outer_report, "overall_crossfit": overall,
         "overall_spearman": overall_rho, "final_cfg": final_cfg,
         "reference_spearman": 0.31,
         "reference": {"fixed": +0.0033, "hand_built": +0.0065, "learned": +0.0052,
                       "shrinkage_piece1": +0.0029}},
        indent=1, default=float), encoding="utf-8")
    print(f"\nfinal cfg={final_cfg} (mode across outer folds), fit on all out-of-fold data")
    print(f"written {args.out} (avpkl-compatible) and {args.report}")


if __name__ == "__main__":
    main()
