#!/usr/bin/env python3
"""Audit R7: the self-improvement curve on a common held-out patient set.

`final_selfimprove.py` evaluated each experience size on whatever patients that
size did not see, so the four points were scored on four different populations
and the 100% point was in-sample. Here every size is trained on a patient
subsample and evaluated on **one common held-out set**, repeated 20 times with
different draws, so the sizes are comparable and the curve carries intervals.

The fixed-strategy reference is evaluated on the same held-out patients in the
same repeats, since the point of the comparison is "does experience beat not
choosing at all on these cases".
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build, STRATEGIES
from q_model import patient_of

SIZES = [0.10, 0.25, 0.50, 1.00]
GATE = 0.90
HELDOUT = 0.25          # fraction of patients reserved, common to all sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_selfimprove.json"))
    args = ap.parse_args()
    R = Path(r"D:\data\reports")

    d = build([R / "strategy_table_oof.csv", R / "strategy6_oof.csv"],
              R / "cand_cv_oof_pos.csv", R / "phase4_diag_model.pkl")
    cfg = json.load(open(R / "phase5_action_value.json", encoding="utf-8"))["cfg"]
    meta, y, X = d["meta"], d["y"], d["X"]
    groups = np.array([patient_of(m["case"]) for m in meta])

    dq = []
    for tab in ("strategy_table_oof.csv", "strategy6_oof.csv"):
        for r in csv.DictReader(open(R / tab, newline="", encoding="utf-8")):
            dq.append(float(r["d_q"]) if r["d_q"] not in ("", "nan") else np.nan)
    dq = np.array(dq, float)

    gated = np.array([m["p_failure"] >= GATE for m in meta])
    idx = collections.defaultdict(list)
    for i, m in enumerate(meta):
        if gated[i]:
            idx[(m["case"], m["component_id"], m["strategy"] == "recover")].append(i)
    print(f"{len(idx)} gated components, {int(gated.sum())} candidate actions")

    pats = np.array(sorted(set(groups)))
    acc = {f: {"R": [], "dq": [], "prec": []} for f in SIZES}
    acc["fixed"] = {"R": [], "dq": [], "prec": []}

    for rep in range(args.repeats):
        rng = np.random.default_rng(rep)
        perm = rng.permutation(pats)
        n_ho = max(1, int(round(HELDOUT * len(pats))))
        ho = set(perm[:n_ho])
        pool = perm[n_ho:]
        in_ho = np.array([g in ho for g in groups])
        # components whose rows are entirely in the held-out set
        keys = [k for k, ix in idx.items() if all(in_ho[i] for i in ix)]
        if not keys:
            continue
        for frac in SIZES:
            k = max(1, int(round(frac * len(pool))))
            tr_pats = set(pool[:k])
            tr = np.array([g in tr_pats for g in groups])
            m = HistGradientBoostingRegressor(random_state=0, **cfg)
            m.fit(X[tr], y[tr])
            pred = m.predict(X)
            sel = [max(idx[kk], key=lambda i: pred[i]) for kk in keys]
            v, q = y[sel], dq[sel]
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            acc[frac]["R"].append(float(np.nanmean(q)))
            acc[frac]["dq"].append(float(np.nanmean(v)))
            acc[frac]["prec"].append(100 * pos / max(pos + neg, 1))
        # fixed reference on the same held-out components
        fx = []
        for kk in keys:
            for i in idx[kk]:
                if meta[i]["strategy"] == "topology_remove":
                    fx.append(i); break
        if fx:
            v, q = y[fx], dq[fx]
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            acc["fixed"]["R"].append(float(np.nanmean(q)))
            acc["fixed"]["dq"].append(float(np.nanmean(v)))
            acc["fixed"]["prec"].append(100 * pos / max(pos + neg, 1))
        if (rep + 1) % 5 == 0:
            print(f"  repeat {rep+1}/{args.repeats}", flush=True)

    def ci(v):
        a = np.array(v, float)
        return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    print(f"\n{'experience':16s}{'R(a)=E[dQ|a]':>26s}{'mean d_qdice':>26s}"
          f"{'precision %':>22s}")
    res = {}
    for key in SIZES + ["fixed"]:
        if not acc[key]["R"]:
            continue
        r, rl, rh = ci(acc[key]["R"])
        q, ql, qh = ci(acc[key]["dq"])
        p, pl, ph = ci(acc[key]["prec"])
        lab = ("fixed (no experience)" if key == "fixed"
               else f"{int(key*100)}% of patients")
        print(f"{lab:16s}{r:+9.5f} [{rl:+.5f},{rh:+.5f}]"
              f"{q:+9.5f} [{ql:+.5f},{qh:+.5f}]"
              f"{p:8.1f} [{pl:.1f},{ph:.1f}]")
        res[str(key)] = {"R_mean": r, "R_lo": rl, "R_hi": rh,
                         "dq_mean": q, "dq_lo": ql, "dq_hi": qh,
                         "prec_mean": p, "prec_lo": pl, "prec_hi": ph,
                         "repeats": len(acc[key]["R"])}
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
