#!/usr/bin/env python3
"""Phase VIII: causal, LCB-gated surgical overrides of the hand-built policy.

See `reports/phase8_causal_override_plan.md`. For each (failure class,
size-bin) cell, find the hand-built policy's own pick per component
(replicating `phase5_agent.py`'s hand-built scoring exactly), compute the
*paired* causal delta of every measured alternative strategy against that
pick, and promote the best alternative only if its lower-confidence bound
excludes zero and its win rate clears a regression guard -- otherwise the
hybrid policy is byte-identical to the hand-built policy. No blending.

The fitted artefact decodes the same 45-column X every Phase VII piece used
and drops into `phase5_agent.py --avpkl` unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build, STRATEGIES
from phase7_shrinkage import FOLDS, assign_bins, bin_index, primary_metric
from phase7_ranking import select_per_component, GATE

MIN_PAIRED = 20
MIN_WIN_RATE = 0.60
NBOOT = 2000
POLICY_PATH = Path(r"D:\data\reports\phase4_policy_noban.json")


class HybridPolicy:
    """Hand-built scoring everywhere, except a promoted (class, size_bin)
    cell's strategy gets a dominant score when it is among the live
    candidates for that component. Scikit-learn-style `.predict(X)`."""

    def __init__(self, policy: dict, overrides: dict, size_edges,
                 feat_names: list[str], cls_names: list[str]):
        self.policy = policy
        self.overrides = overrides  # {(class, size_bin): strategy}
        self.size_edges = size_edges
        self.i_logvol = feat_names.index("log_volume")
        self.i_strat0 = len(feat_names) + 1  # +1 for p_failure column
        self.strategies = list(STRATEGIES)
        self.i_cls0 = self.i_strat0 + len(self.strategies)
        self.cls_names = cls_names

    def _size_bin(self, v):
        if not np.isfinite(v):
            return 1
        lo, hi = self.size_edges
        return 0 if v < lo else (1 if v < hi else 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            sbin = self._size_bin(row[self.i_logvol])
            strat_oh = row[self.i_strat0:self.i_strat0 + len(self.strategies)]
            strat = self.strategies[int(np.argmax(strat_oh))]
            cls_oh = row[self.i_cls0:]
            cls = ("missed" if not cls_oh.any()
                  else self.cls_names[int(np.argmax(cls_oh))])
            order = self.policy.get(cls, [])
            base = (len(order) - order.index(strat)) if strat in order else -1.0
            if self.overrides.get((cls, sbin)) == strat:
                out[i] = 1000.0  # dominant: wins whenever it is a live candidate
            else:
                out[i] = base
        return out


def handbuilt_pick(order: list, strats_here: dict) -> str | None:
    """`strats_here`: {strategy: y}. The hand-built policy's pick among the
    strategies actually measured for this component, or None."""
    for s in order:
        if s in strats_here:
            return s
    return None


def paired_bootstrap_delta(pairs: list, seed=2026):
    a = np.array([p[0] for p in pairs])  # alt
    b = np.array([p[1] for p in pairs])  # handbuilt
    n = len(a)
    rng = np.random.default_rng(seed)
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, n, n)
        boot[i] = np.mean(a[ix] - b[ix])
    lo = float(np.percentile(boot, 2.5))
    win_rate = float((a > b).mean())
    return {"n": n, "mean": float(np.mean(a - b)), "lo": lo,
            "win_rate": win_rate}


def find_overrides(cls_arr, strat_arr, sbin_arr, y_arr, comp_key_arr,
                    idx: np.ndarray, policy: dict) -> dict:
    """Search cells on the rows in `idx` only (training data)."""
    by_comp = collections.defaultdict(dict)
    comp_cls = {}
    for i in idx:
        by_comp[comp_key_arr[i]][strat_arr[i]] = y_arr[i]
        comp_cls[comp_key_arr[i]] = (cls_arr[i], sbin_arr[i])

    cell_pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, strats_here in by_comp.items():
        cls, sbin = comp_cls[key]
        order = policy.get(cls, [])
        pick = handbuilt_pick(order, strats_here)
        if pick is None:
            continue
        y_pick = strats_here[pick]
        for s, y in strats_here.items():
            if s == pick:
                continue
            cell_pairs[(cls, sbin)][s].append((y, y_pick))

    overrides, report = {}, []
    for (cls, sbin), alt_dict in cell_pairs.items():
        best_alt, best_stat = None, None
        for alt, pairs in alt_dict.items():
            if len(pairs) < MIN_PAIRED:
                continue
            stat = paired_bootstrap_delta(pairs)
            if best_stat is None or stat["mean"] > best_stat["mean"]:
                best_alt, best_stat = alt, stat
        if best_alt is None:
            continue
        promoted = best_stat["lo"] > 0 and best_stat["win_rate"] >= MIN_WIN_RATE
        report.append({"class": cls, "size_bin": sbin, "alt": best_alt,
                        **best_stat, "promoted": promoted})
        if promoted:
            overrides[(cls, sbin)] = best_alt
    return overrides, report


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
                    default=Path(r"D:\data\reports\phase8_causal_override.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase8_causal_override.json"))
    args = ap.parse_args()

    policy = json.loads(POLICY_PATH.read_text())["policy"]
    d = build(args.table, args.cand, args.diagpkl)
    meta, y, X, names = d["meta"], d["y"], d["X"], d["names"]
    n = len(meta)
    cls_arr = np.array([m["pred_class"] for m in meta], dtype=object)
    strat_arr = np.array([m["strategy"] for m in meta], dtype=object)
    pfail_arr = np.array([m["p_failure"] for m in meta], dtype=float)
    comp_key_arr = np.empty(n, dtype=object)
    for i, m in enumerate(meta):
        comp_key_arr[i] = (m["case"], m["component_id"], m["strategy"] == "recover")
    i_logvol = names.index("log_volume")
    logvol_arr = X[:, i_logvol]

    splits = json.loads(args.splits.read_text())
    owner = {}
    for k in FOLDS:
        for s in splits[k]["val"]:
            owner[s] = k
    fold_arr = np.array([owner.get(m["case"], -1) for m in meta])
    print(f"{n} actions, {int((fold_arr < 0).sum())} without a Phase IIIb fold")

    outer_report = []
    all_chosen: dict = {}
    for outer in FOLDS:
        te = fold_arr == outer
        tr = np.where(~te & (fold_arr >= 0))[0]
        sb_e, _ = assign_bins(logvol_arr[tr], logvol_arr[tr])
        sbin_arr = np.array([bin_index(v, sb_e) for v in logvol_arr])
        overrides, cells = find_overrides(cls_arr, strat_arr, sbin_arr, y,
                                           comp_key_arr, tr, policy)
        model = HybridPolicy(policy, overrides, sb_e, names[:26], d["pred_classes"])
        score_all = model.predict(X)
        chosen = select_per_component(score_all, comp_key_arr, pfail_arr, te,
                                       gate=GATE)
        pm = primary_metric(y, chosen) if chosen else {"n_selected": 0,
                                                        "mean_d_qdice": 0.0,
                                                        "precision": 0.0}
        promoted_cells = [c for c in cells if c["promoted"]]
        print(f"  outer fold {outer}: {len(promoted_cells)} promoted cells "
              f"of {len(cells)} searched  held-out n={pm['n_selected']}  "
              f"mean d_qdice={pm['mean_d_qdice']:+.5f}  precision={pm['precision']:.1f}%")
        for c in promoted_cells:
            print(f"    {c['class']:22s} size_bin={c['size_bin']}  -> {c['alt']:16s}  "
                  f"n={c['n']:3d}  delta={c['mean']:+.4f}  LCB={c['lo']:+.4f}  "
                  f"win_rate={c['win_rate']:.2f}")
        outer_report.append({"fold": outer, "cells": cells, **pm})
        all_chosen.update(chosen)

    overall = primary_metric(y, all_chosen) if all_chosen else {
        "n_selected": 0, "mean_d_qdice": 0.0, "precision": 0.0}
    print(f"\ncross-fitted overall: n={overall['n_selected']}  "
          f"mean d_qdice={overall['mean_d_qdice']:+.5f}  precision={overall['precision']:.1f}%")

    # ---- final production artefact: promote cells from ALL out-of-fold data
    mask = np.where(fold_arr >= 0)[0]
    sb_e, _ = assign_bins(logvol_arr[mask], logvol_arr[mask])
    sbin_arr_final = np.array([bin_index(v, sb_e) for v in logvol_arr])
    overrides, cells = find_overrides(cls_arr, strat_arr, sbin_arr_final, y,
                                       comp_key_arr, mask, policy)
    promoted = [c for c in cells if c["promoted"]]
    print(f"\nfinal: {len(promoted)} promoted cells (fit on all out-of-fold data)")
    for c in promoted:
        print(f"  {c['class']:22s} size_bin={c['size_bin']}  -> {c['alt']:16s}  "
              f"n={c['n']:3d}  delta={c['mean']:+.4f}  LCB={c['lo']:+.4f}  "
              f"win_rate={c['win_rate']:.2f}")
    final_model = HybridPolicy(policy, overrides, sb_e, names[:26], d["pred_classes"])
    with open(args.out, "wb") as fh:
        pickle.dump({"model": final_model, "pred_classes": list(d["pred_classes"])},
                    fh)
    args.report.write_text(json.dumps(
        {"outer_folds": outer_report, "overall_crossfit": overall,
         "final_cells": cells, "final_promoted": promoted,
         "reference": {"fixed": +0.0033, "hand_built": +0.0065, "learned": +0.0052}},
        indent=1, default=float), encoding="utf-8")
    print(f"\nwritten {args.out} (avpkl-compatible) and {args.report}")


if __name__ == "__main__":
    main()
