#!/usr/bin/env python3
"""Phase V Experiment 5: a learned action-value model.

Replaces the Phase IV hand-built policy, which picked one strategy per failure
class by mean Delta-Dice over the measured table. That policy cannot condition on
the component in front of it: every `spurious_other` component gets the same
strategy whatever its volume, SUV or uncertainty. This fits

    d_qdice  ~  f(26 observable features, proposed strategy, p_failure,
                  predicted failure class)

over the 34,999 measured actions, so the choice of strategy can depend on the
component.

Target is `d_qdice`, not `d_dice`: on a tumour-free case Dice is 0/0 the moment a
strategy empties the mask, which is exactly the outcome removal exists to produce.
`d_qdice` scores that as 1.0 (Phase IV section 3); `d_dice` is nan there and is
kept only for the primary endpoint.

Nothing here touches validation or test. Hyperparameters come from a
grouped-by-patient CV search on out-of-fold alone, as in Phase IIIb's corrected
procedure.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_agent import observable_index, observable_rows
from phase4_diag_model import PRED_CLASSES
from q_model import patient_of

STRATEGIES = ["topology_remove", "topology_fill", "topology_merge", "boundary",
              "suv41", "suv_percist", "volume_grow", "volume_shrink",
              "recover", "roi_repredict"]


def diagnose(obs: dict, model: dict, case: str):
    """p_failure and the predicted class for every component of one case.

    Returns {component_id: (p_failure, class_name, probability_vector)}.
    """
    ids, X = observable_rows(obs, case, model["feats"], model["median"])
    if not ids:
        return {}
    classes = list(model["models"].keys())
    P = np.column_stack([model["models"][c].predict_proba(X)[:, 1]
                         for c in classes])
    out = {}
    for i, cid in enumerate(ids):
        j = int(np.argmax(P[i]))
        out[cid] = (float(P[i, j]), classes[j], P[i])
    return out


def build(tables: list[Path], cand: Path, diagpkl: Path) -> dict:
    """Join measured actions to the observable features of their component."""
    D = pickle.load(open(diagpkl, "rb"))
    obs_pred = observable_index(cand, "pred")
    obs_rec = observable_index(cand, "recover")

    # one diagnosis pass per case, reused by every action on that case
    cases = sorted(set(obs_pred) | set(obs_rec))
    diag_pred, diag_rec = {}, {}
    for c in cases:
        diag_pred[c] = diagnose(obs_pred, D["pred"], c)
        diag_rec[c] = diagnose(obs_rec, D["recover"], c)

    feats = D["pred"]["feats"]
    pred_classes = list(D["pred"]["models"].keys())

    # raw feature rows, keyed for the join
    def index_raw(obs, model):
        out = {}
        for c in cases:
            ids, X = observable_rows(obs, c, model["feats"], model["median"])
            for i, cid in enumerate(ids):
                out[(c, cid)] = X[i]
        return out

    raw_pred = index_raw(obs_pred, D["pred"])
    raw_rec = index_raw(obs_rec, D["recover"])

    rows, y, groups, meta = [], [], [], []
    missed = 0
    for tab in tables:
        for r in csv.DictReader(open(tab, newline="", encoding="utf-8")):
            case, strat = r["case"], r["strategy"]
            try:
                cid = int(r["component_id"])
            except ValueError:
                missed += 1
                continue
            is_rec = strat == "recover"
            # the strategy table negates recovery ids to keep them in their own
            # id space; the candidate CSV stores them positive. Without the abs
            # every recover row silently fails the join -- which is all 1,086 of
            # them, and with them the only strategy addressing `missed`, the
            # largest oracle headroom in the whole diagnosis.
            if is_rec:
                cid = abs(cid)
            raw = raw_rec if is_rec else raw_pred
            dg = (diag_rec if is_rec else diag_pred).get(case, {})
            key = (case, cid)
            if key not in raw or cid not in dg:
                missed += 1
                continue
            try:
                target = float(r["d_qdice"])
            except (ValueError, KeyError):
                missed += 1
                continue
            if not np.isfinite(target):
                missed += 1
                continue

            pfail, pcls, _ = dg[cid]
            x = list(raw[key])
            x.append(pfail)
            x.extend(1.0 if s == strat else 0.0 for s in STRATEGIES)
            x.extend(1.0 if c == pcls else 0.0 for c in pred_classes)
            rows.append(x)
            y.append(target)
            groups.append(patient_of(case))
            meta.append({"case": case, "component_id": cid, "strategy": strat,
                         "p_failure": pfail, "pred_class": pcls,
                         "d_dice": r["d_dice"], "classes": r["classes"]})

    names = (list(feats) + ["p_failure"]
             + [f"strat_{s}" for s in STRATEGIES]
             + [f"cls_{c}" for c in pred_classes])
    X = np.array(rows, float)
    print(f"{len(X)} actions joined, {missed} dropped (no feature row, "
          f"no diagnosis, or nan target)")
    print(f"{X.shape[1]} features = {len(feats)} observable + 1 p_failure "
          f"+ {len(STRATEGIES)} strategy + {len(pred_classes)} class")
    return {"X": X, "y": np.array(y, float), "groups": np.array(groups),
            "names": names, "meta": meta, "pred_classes": pred_classes}


GRID = [{"max_depth": d, "learning_rate": lr, "max_iter": it,
         "min_samples_leaf": leaf, "l2_regularization": l2}
        for d in (3, 5, None)
        for lr in (0.03, 0.06, 0.1)
        for it in (200, 400)
        for leaf in (20, 50)
        for l2 in (0.0, 1.0)]


def search(d: dict, folds: int = 5, grid=GRID, seed: int = 0) -> dict:
    """Grouped-by-patient CV over the grid. Scored by Spearman, not R-squared.

    Phase IV's lesson is that what matters is ranking actions against each other,
    not predicting the magnitude of an effect whose median is 0.0002.
    """
    from scipy.stats import spearmanr
    gkf = GroupKFold(n_splits=folds)
    splits = list(gkf.split(d["X"], d["y"], d["groups"]))
    best, rows = None, []
    for i, cfg in enumerate(grid):
        rho = []
        for tr, te in splits:
            m = HistGradientBoostingRegressor(random_state=seed, **cfg)
            m.fit(d["X"][tr], d["y"][tr])
            p = m.predict(d["X"][te])
            r = spearmanr(p, d["y"][te]).statistic
            rho.append(0.0 if not np.isfinite(r) else r)
        s = float(np.mean(rho))
        rows.append({"cfg": cfg, "spearman": s})
        if best is None or s > best["spearman"]:
            best = {"cfg": cfg, "spearman": s}
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(grid)} best Spearman {best['spearman']:.4f}",
                  flush=True)
    return {"best": best, "all": rows}


def oof_predictions(d: dict, cfg: dict, folds: int = 5, seed: int = 0):
    """Out-of-fold predictions, each from the fold model that did not see it."""
    gkf = GroupKFold(n_splits=folds)
    pred = np.full(len(d["y"]), np.nan)
    for tr, te in gkf.split(d["X"], d["y"], d["groups"]):
        m = HistGradientBoostingRegressor(random_state=seed, **cfg)
        m.fit(d["X"][tr], d["y"][tr])
        pred[te] = m.predict(d["X"][te])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, nargs="+", required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\phase5_action_value.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\data\reports\phase5_action_value.json"))
    ap.add_argument("--quick", action="store_true",
                    help="a 6-config grid, for wiring checks only")
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    grid = GRID[:6] if args.quick else GRID
    print(f"grid search: {len(grid)} configs, grouped by patient")
    res = search(d, grid=grid)
    cfg = res["best"]["cfg"]
    print(f"best {cfg} Spearman {res['best']['spearman']:.4f}")

    pred = oof_predictions(d, cfg)
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred, d["y"]).statistic)
    print(f"out-of-fold Spearman {rho:.4f}")

    final = HistGradientBoostingRegressor(random_state=0, **cfg)
    final.fit(d["X"], d["y"])
    with open(args.out, "wb") as fh:
        pickle.dump({"model": final, "names": d["names"], "cfg": cfg,
                     "strategies": STRATEGIES,
                     "pred_classes": d["pred_classes"],
                     "feats": pickle.load(open(args.diagpkl, "rb"))
                     ["pred"]["feats"]}, fh)
    np.save(args.out.with_suffix(".oofpred.npy"), pred)
    with open(args.out.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(d["meta"], fh)
    args.report.write_text(json.dumps(
        {"n": len(d["y"]), "cfg": cfg, "cv_spearman": res["best"]["spearman"],
         "oof_spearman": rho}, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
