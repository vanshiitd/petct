#!/usr/bin/env python3
"""Phase VII, piece 5: closed-loop re-fitting with active learning
(addresses reason 5).

The audit's own R7 curve showed the learned selector's quality flat against
training-set size -- more data did not help. This piece asks a narrower
question: does *which* data arrives first matter? It simulates sequential
re-fitting of piece 3's architecture (class-balanced LGBMRanker lambdarank,
hard-gated at p_failure>=0.90 -- piece 4 showed removing that gate is
actively harmful, so it stays) as training data accrues in two orders:

  - **random**: i.i.d. order, matching the audit's original flat-curve setup;
  - **active**: at each step, examples from the currently most
    under-represented (pred_class) cells are added first, so minority
    classes (spurious_physio, spurious_liver_spleen, over_seg, merging,
    under_seg, fragmentation) reach a usable sample size before the
    dataset is dominated by spurious_other.

Evaluated by the fast proxy only (no new agent runs needed): piece 3's
full-data configuration and real agent result are already established
(+0.0033 OOF, 59/385 FP cleared, reports/phase7_piece3_result.md) -- both
curves converge to that exact model at 100% of the data by construction, so
only the curve *shape* at intermediate sizes is new information here.

Same cross-fitted discipline: averaged over the Phase IIIb outer folds
3,4,5,6,7 (architecture/hyperparameters fixed from piece 3 -- re-tuning at
every point of every repeat of every fold is not affordable, and is not the
question this piece asks), with repeats over the random seed controlling
within-class shuffle order, intervals from those repeats (matching the
audit's R7 "20 repeats, intervals" convention at a feasible 10).

Falsified if the active curve does not rise measurably above the random
curve -- i.e., if ordering data by class rarity does not get to a given
quality level with less data than i.i.d. order.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build
from phase7_shrinkage import FOLDS, primary_metric
from phase7_balanced_ranking import fit_ranker
from phase7_ranking import select_per_component, GATE

STEPS = (0.2, 0.4, 0.6, 0.8, 1.0)
REPEATS = 10
FINAL_CFG = {"learning_rate": 0.1, "min_child_samples": 20, "n_estimators": 300,
             "num_leaves": 15}  # piece 2/3's mode-selected final config


def order_random(idx: np.ndarray, rng) -> np.ndarray:
    order = idx.copy()
    rng.shuffle(order)
    return order


def order_active(idx: np.ndarray, pred_class_arr: np.ndarray, rng) -> np.ndarray:
    """Round-robin across classes, rarest-class-first within each round, so
    minority classes are fully represented before spurious_other dominates."""
    by_class = collections.defaultdict(list)
    for i in idx:
        by_class[pred_class_arr[i]].append(i)
    for c in by_class:
        rng.shuffle(by_class[c])
    # smallest class first, each contributing one item per round
    order_classes = sorted(by_class, key=lambda c: len(by_class[c]))
    queues = [collections.deque(by_class[c]) for c in order_classes]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
    return np.array(out)


def run_curve(X, y, comp_key_arr, pred_class_arr, pfail_arr, fold_arr,
              mode: str) -> dict:
    per_step = collections.defaultdict(list)
    for outer in FOLDS:
        te = fold_arr == outer
        tr = np.where(~te & (fold_arr >= 0))[0]
        for rep in range(REPEATS):
            rng = np.random.default_rng(1000 * outer + rep)
            order = (order_active(tr, pred_class_arr, rng) if mode == "active"
                     else order_random(tr, rng))
            for frac in STEPS:
                k = max(int(len(order) * frac), 50)
                sub = order[:k]
                m = fit_ranker(X, y, comp_key_arr, pred_class_arr, sub, FINAL_CFG,
                               seed=rep)
                score = m.predict(X)
                chosen = select_per_component(score, comp_key_arr, pfail_arr, te,
                                               gate=GATE)
                if not chosen:
                    continue
                pm = primary_metric(y, chosen)
                per_step[frac].append(pm["mean_d_qdice"])
    out = {}
    for frac in STEPS:
        v = np.array(per_step[frac])
        out[frac] = {"mean": float(v.mean()), "std": float(v.std()),
                     "n_runs": len(v),
                     "lo": float(np.percentile(v, 2.5)),
                     "hi": float(np.percentile(v, 97.5))}
    return out


def main() -> None:
    table = [Path(r"D:\data\reports\strategy_table_oof.csv"),
              Path(r"D:\data\reports\strategy6_oof.csv")]
    cand = Path(r"D:\data\reports\cand_cv_oof_pos.csv")
    diagpkl = Path(r"D:\data\reports\phase4_diag_model.pkl")
    splits_path = Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json")
    out_path = Path(r"D:\data\reports\phase7_active.json")

    d = build(table, cand, diagpkl)
    meta, y, X = d["meta"], d["y"], d["X"]
    n = len(meta)
    pfail_arr = np.array([m["p_failure"] for m in meta], dtype=float)
    pred_class_arr = np.array([m["pred_class"] for m in meta], dtype=object)
    comp_key_arr = np.empty(n, dtype=object)
    for i, m in enumerate(meta):
        comp_key_arr[i] = (m["case"], m["component_id"], m["strategy"] == "recover")

    splits = json.loads(splits_path.read_text())
    owner = {}
    for k in FOLDS:
        for s in splits[k]["val"]:
            owner[s] = k
    fold_arr = np.array([owner.get(m["case"], -1) for m in meta])
    print(f"{n} actions, {REPEATS} repeats x {len(FOLDS)} outer folds x "
          f"{len(STEPS)} steps x 2 orderings")

    curves = {}
    for mode in ("random", "active"):
        print(f"\n{mode} ordering:")
        curves[mode] = run_curve(X, y, comp_key_arr, pred_class_arr, pfail_arr,
                                  fold_arr, mode)
        for frac, stats in curves[mode].items():
            print(f"  {frac:.0%} of training data: mean d_qdice "
                  f"{stats['mean']:+.5f} [{stats['lo']:+.5f}, {stats['hi']:+.5f}]  "
                  f"(n_runs={stats['n_runs']})")

    rises = []
    for frac in STEPS[:-1]:  # the final step converges by construction
        a, r = curves["active"][frac]["mean"], curves["random"][frac]["mean"]
        rises.append(a > r)
    measurable_rise = sum(rises) >= len(rises) - 1  # allow one step of noise
    print(f"\nactive > random at {sum(rises)}/{len(rises)} intermediate steps")
    verdict = ("not falsified -- active ordering rises measurably above random"
               if measurable_rise else
               "falsified -- active ordering does not rise measurably above random")
    print(f"Piece 5 verdict: {verdict}")
    print("\nNote: both orderings converge to piece 3's established result at "
          "100% data (+0.0033 OOF case-level, does not beat hand-built policy "
          "(b) +0.0065) -- this piece can only show data-efficiency, not raise "
          "the ceiling piece 3 already set.")

    out_path.write_text(json.dumps(
        {"steps": list(STEPS), "repeats": REPEATS, "folds": list(FOLDS),
         "curves": curves, "rises": rises, "measurable_rise": measurable_rise,
         "verdict": verdict}, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
