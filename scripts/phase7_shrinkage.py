#!/usr/bin/env python3
"""Phase VII, piece 1: a hierarchical empirical-Bayes shrinkage policy.

Reason 4 from the retrospective: the hand-built policy (a flat per-class mean
outcome) beat the learned action-value regressor, cross-fitted, at every budget.
That is the signature of a low-variance baseline beating a high-variance model
on a weak-signal problem. The fix is not "more model", it is "more
conditioning, but only where the data can support it" -- partial pooling.

Hierarchy, coarsest to finest:

    level 0  global mean outcome per strategy
    level 1  (failure class, strategy)                     <- the hand-built policy's own granularity
    level 2  (class, size-bin, strategy)
    level 3  (class, size-bin, SUV-bin, strategy)

Each level's estimate is a precision-weighted blend of its own raw mean and its
parent's already-shrunk estimate:

    est_L(cell) = (n_cell * raw_mean_cell + k * est_{L-1}(parent)) / (n_cell + k)

As k -> infinity this collapses to level 1 (the hand-built policy, exactly).
As k -> 0 it collapses to the raw finest-grain mean (which is what overfit the
original learned model). k is selected by **nested** cross-validation: for each
outer fold (one of the Phase IIIb folds 3,4,5,6,7), k is chosen using only the
other four folds (inner leave-one-fold-out), never the held-out fold, never
validation, never test.

The policy target is `d_qdice`, exactly as in `phase5_action_value.py`: it
scores an emptied tumour-free mask as 1.0, which is what the loop is actually
optimising for Dice-wise (`d_dice` is undefined there).

The fitted artefact is a `ShrinkagePolicy` instance with a scikit-learn-style
`.predict(X)` method, so it drops into `phase5_agent.py --avpkl` with **zero**
changes to the agent: `X` is exactly the 45-column matrix
`phase5_action_value.build()` already constructs (26 raw features, p_failure,
10-way strategy one-hot, 8-way class one-hot).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build, STRATEGIES
from q_model import patient_of

FOLDS = (3, 4, 5, 6, 7)
K_GRID = [1, 2, 5, 10, 20, 40, 80, 160, 320, 1000]
SIZE_Q = (1 / 3, 2 / 3)   # tertile edges for the size bin
SUV_Q = (1 / 3, 2 / 3)    # tertile edges for the SUV bin
GATE = 0.90               # matches the deployed p_failure gate throughout


class ShrinkagePolicy:
    """Hierarchical empirical-Bayes lookup. Scikit-learn-compatible `.predict`.

    `table` is {(class, size_bin, suv_bin, strategy): shrunk_estimate}, already
    fully resolved (no further blending needed at inference) and exhaustive
    over every (class, size_bin, suv_bin, strategy) combination that appeared
    in training, plus a `fallback` level-0-per-strategy dict for anything that
    did not.
    """

    def __init__(self, table: dict, fallback: dict, size_edges, suv_edges,
                feat_names: list[str]):
        self.table = table
        self.fallback = fallback
        self.size_edges = size_edges
        self.suv_edges = suv_edges
        self.i_logvol = feat_names.index("log_volume")
        self.i_suv = feat_names.index("suv_max_over_liver")
        self.i_strat0 = len(feat_names) + 1  # +1 for p_failure column
        self.strategies = list(STRATEGIES)
        n_strat = len(self.strategies)
        self.i_cls0 = self.i_strat0 + n_strat

    def _bin(self, v, edges):
        if not np.isfinite(v):
            return 1  # middle bin for missing/degenerate values
        lo, hi = edges
        return 0 if v < lo else (1 if v < hi else 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            logvol = row[self.i_logvol]
            suv = row[self.i_suv]
            sbin = self._bin(logvol, self.size_edges)
            ubin = self._bin(suv, self.suv_edges)
            strat_oh = row[self.i_strat0:self.i_strat0 + len(self.strategies)]
            strat = self.strategies[int(np.argmax(strat_oh))]
            cls_oh = row[self.i_cls0:]
            cls = ("missed" if not cls_oh.any()
                  else self._cls_names[int(np.argmax(cls_oh))])
            key = (cls, sbin, ubin, strat)
            out[i] = self.table.get(key, self.fallback.get(strat, 0.0))
        return out


def assign_bins(logvol: np.ndarray, suv: np.ndarray):
    """Tertile edges fit on the given (training) data only."""
    finite_v = logvol[np.isfinite(logvol)]
    finite_s = suv[np.isfinite(suv)]
    size_edges = tuple(np.quantile(finite_v, SIZE_Q)) if len(finite_v) else (0, 0)
    suv_edges = tuple(np.quantile(finite_s, SUV_Q)) if len(finite_s) else (0, 0)
    return size_edges, suv_edges


def bin_index(v: float, edges) -> int:
    if not np.isfinite(v):
        return 1
    lo, hi = edges
    return 0 if v < lo else (1 if v < hi else 2)


def fit_table(cls_arr, strat_arr, sbin_arr, ubin_arr, y_arr, k: float) -> dict:
    """One hierarchical partial-pooling fit. Pure groupby aggregation - cheap."""
    import collections
    # level 0: global per-strategy mean
    l0 = {}
    for s in set(strat_arr):
        m = strat_arr == s
        l0[s] = float(np.nanmean(y_arr[m])) if m.any() else 0.0

    # level 1: (class, strategy), shrunk toward level 0
    sums1 = collections.defaultdict(lambda: [0.0, 0])
    for c, s, y in zip(cls_arr, strat_arr, y_arr):
        if not np.isfinite(y):
            continue
        sums1[(c, s)][0] += y
        sums1[(c, s)][1] += 1
    l1 = {}
    for (c, s), (tot, n) in sums1.items():
        raw = tot / n
        l1[(c, s)] = (n * raw + k * l0[s]) / (n + k)

    # level 2: (class, size_bin, strategy), shrunk toward level 1
    sums2 = collections.defaultdict(lambda: [0.0, 0])
    for c, s, sb, y in zip(cls_arr, strat_arr, sbin_arr, y_arr):
        if not np.isfinite(y):
            continue
        sums2[(c, sb, s)][0] += y
        sums2[(c, sb, s)][1] += 1
    l2 = {}
    for (c, sb, s), (tot, n) in sums2.items():
        raw = tot / n
        parent = l1.get((c, s), l0[s])
        l2[(c, sb, s)] = (n * raw + k * parent) / (n + k)

    # level 3: (class, size_bin, suv_bin, strategy), shrunk toward level 2
    sums3 = collections.defaultdict(lambda: [0.0, 0])
    for c, s, sb, ub, y in zip(cls_arr, strat_arr, sbin_arr, ubin_arr, y_arr):
        if not np.isfinite(y):
            continue
        sums3[(c, sb, ub, s)][0] += y
        sums3[(c, sb, ub, s)][1] += 1
    l3 = {}
    for (c, sb, ub, s), (tot, n) in sums3.items():
        raw = tot / n
        parent = l2.get((c, sb, s), l1.get((c, s), l0[s]))
        l3[(c, sb, ub, s)] = (n * raw + k * parent) / (n + k)

    return {"l0": l0, "table": l3}


def select_per_component(table: dict, fallback: dict, cls_arr, strat_arr,
                         sbin_arr, ubin_arr, comp_key_arr, pfail_arr,
                         gate: float = GATE):
    """Argmax strategy per gated component; returns {comp_key: (strategy, idx)}."""
    import collections
    cands = collections.defaultdict(list)
    for i in range(len(cls_arr)):
        if pfail_arr[i] < gate:
            continue
        cands[comp_key_arr[i]].append(i)
    chosen = {}
    for key, idxs in cands.items():
        best_i, best_v = None, -np.inf
        for i in idxs:
            c, s, sb, ub = cls_arr[i], strat_arr[i], sbin_arr[i], ubin_arr[i]
            v = table.get((c, sb, ub, s), fallback.get(s, -np.inf))
            if v > best_v:
                best_v, best_i = v, i
        chosen[key] = best_i
    return chosen


def primary_metric(y_arr, chosen: dict) -> dict:
    idx = np.array(list(chosen.values()))
    v = y_arr[idx]
    v = v[np.isfinite(v)]
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    return {"n_selected": len(idx), "mean_d_qdice": float(np.nanmean(v)),
            "precision": 100 * pos / max(pos + neg, 1)}


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
                    default=Path(r"D:\data\reports\phase7_shrinkage.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase7_shrinkage.json"))
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    meta, y, X, names = d["meta"], d["y"], d["X"], d["names"]
    n = len(meta)
    cls_arr = np.array([m["pred_class"] for m in meta], dtype=object)
    strat_arr = np.array([m["strategy"] for m in meta], dtype=object)
    pfail_arr = np.array([m["p_failure"] for m in meta], dtype=float)
    # Built elementwise into a pre-allocated object array, not via
    # np.array(list_of_equal_length_tuples): numpy silently broadcasts the
    # latter into a 2D array (one column per tuple slot) instead of an array
    # of tuple objects, which makes every element unhashable.
    comp_key_arr = np.empty(n, dtype=object)
    for i, m in enumerate(meta):
        comp_key_arr[i] = (m["case"], m["component_id"],
                          m["strategy"] == "recover")
    i_logvol, i_suv = names.index("log_volume"), names.index("suv_max_over_liver")
    logvol_arr, suv_arr = X[:, i_logvol], X[:, i_suv]

    splits = json.loads(args.splits.read_text())
    owner = {}
    for k in FOLDS:
        for s in splits[k]["val"]:
            owner[s] = k
    fold_arr = np.array([owner.get(m["case"], -1) for m in meta])
    print(f"{n} actions, {int((fold_arr < 0).sum())} without a Phase IIIb fold")

    # ---- outer cross-fitted loop ------------------------------------------
    all_chosen = {}
    outer_report = []
    for outer in FOLDS:
        te = fold_arr == outer
        tr = ~te & (fold_arr >= 0)
        # inner CV: leave-one-(training)-fold-out, over the 4 remaining folds
        inner_folds = [f for f in FOLDS if f != outer]
        best_k, best_score = None, -np.inf
        for k in K_GRID:
            scores = []
            for held in inner_folds:
                itr = tr & (fold_arr != held)
                ite = tr & (fold_arr == held)
                if itr.sum() < 500 or ite.sum() < 50:
                    continue
                sb_e, su_e = assign_bins(logvol_arr[itr], suv_arr[itr])
                sb = np.array([bin_index(v, sb_e) for v in logvol_arr[itr]])
                su = np.array([bin_index(v, su_e) for v in suv_arr[itr]])
                fit = fit_table(cls_arr[itr], strat_arr[itr], sb, su, y[itr], k)
                sb_te = np.array([bin_index(v, sb_e) for v in logvol_arr[ite]])
                su_te = np.array([bin_index(v, su_e) for v in suv_arr[ite]])
                chosen = select_per_component(
                    fit["table"], fit["l0"], cls_arr[ite], strat_arr[ite],
                    sb_te, su_te, comp_key_arr[ite], pfail_arr[ite])
                if not chosen:
                    continue
                m = primary_metric(y[ite], chosen)
                scores.append(m["mean_d_qdice"])
            if scores:
                s = float(np.mean(scores))
                if s > best_score:
                    best_score, best_k = s, k
        # fit final table for this outer fold on all 4 training folds
        sb_e, su_e = assign_bins(logvol_arr[tr], suv_arr[tr])
        sb = np.array([bin_index(v, sb_e) for v in logvol_arr[tr]])
        su = np.array([bin_index(v, su_e) for v in suv_arr[tr]])
        fit = fit_table(cls_arr[tr], strat_arr[tr], sb, su, y[tr], best_k)
        sb_te = np.array([bin_index(v, sb_e) for v in logvol_arr[te]])
        su_te = np.array([bin_index(v, su_e) for v in suv_arr[te]])
        chosen_local = select_per_component(
            fit["table"], fit["l0"], cls_arr[te], strat_arr[te],
            sb_te, su_te, comp_key_arr[te], pfail_arr[te])
        m = primary_metric(y[te], chosen_local)
        print(f"  outer fold {outer}: inner-selected k={best_k}  "
              f"held-out n={m['n_selected']}  mean d_qdice={m['mean_d_qdice']:+.5f}  "
              f"precision={m['precision']:.1f}%")
        outer_report.append({"fold": outer, "k": best_k, **m})
        # chosen_local's values are indices into the HELD-OUT FOLD'S sliced
        # arrays (0..len(te_idx)-1), not into the global y/meta arrays. Remap
        # through te_idx before merging, or every fold after the first
        # silently overwrites with the wrong rows -- same global index space,
        # different underlying cases.
        te_idx = np.where(te)[0]
        chosen = {key: te_idx[local_i] for key, local_i in chosen_local.items()}
        all_chosen.update(chosen)

    overall = primary_metric(y, all_chosen)
    print(f"\ncross-fitted overall: n={overall['n_selected']}  "
          f"mean d_qdice={overall['mean_d_qdice']:+.5f}  "
          f"precision={overall['precision']:.1f}%")
    print("reference (audit, cross-fitted, T=10, gated, same metric):")
    print("  (a) fixed topology_remove        +0.0033  [-0.0035, +0.0099]")
    print("  (b) hand-built policy             +0.0065  [+0.0025, +0.0118]")
    print("  (c) learned action-value model    +0.0052  [+0.0022, +0.0099]")

    # ---- final production artefact: fit on ALL out-of-fold data -----------
    # k chosen by the mode of the 5 outer-fold inner-selected k's -- the most
    # frequently-preferred shrinkage strength across folds -- rather than by
    # looking at validation or test.
    import collections as _c
    k_counts = _c.Counter(o["k"] for o in outer_report)
    final_k = k_counts.most_common(1)[0][0]
    mask = fold_arr >= 0
    sb_e, su_e = assign_bins(logvol_arr[mask], suv_arr[mask])
    sb = np.array([bin_index(v, sb_e) for v in logvol_arr[mask]])
    su = np.array([bin_index(v, su_e) for v in suv_arr[mask]])
    fit = fit_table(cls_arr[mask], strat_arr[mask], sb, su, y[mask], final_k)
    policy = ShrinkagePolicy(fit["table"], fit["l0"], sb_e, su_e, names[:26])
    policy._cls_names = list(d["pred_classes"])
    with open(args.out, "wb") as fh:
        # phase5_agent.py's value() also reads self.av["pred_classes"] to
        # build the class one-hot segment of x, independent of what the
        # policy object does internally -- must be present or value() throws.
        pickle.dump({"model": policy, "pred_classes": list(d["pred_classes"])},
                    fh)
    args.report.write_text(json.dumps(
        {"outer_folds": outer_report, "overall_crossfit": overall,
         "final_k": final_k, "size_edges": sb_e, "suv_edges": su_e,
         "reference": {"fixed": +0.0033, "hand_built": +0.0065, "learned": +0.0052}},
        indent=1, default=float), encoding="utf-8")
    print(f"\nfinal k={final_k} (mode across outer folds), fit on all out-of-fold data")
    print(f"written {args.out} (avpkl-compatible) and {args.report}")


if __name__ == "__main__":
    main()
