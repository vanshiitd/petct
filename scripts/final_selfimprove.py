#!/usr/bin/env python3
"""Work plan section 24: the self-improvement curve.

Does the strategy selector get better as it accumulates experience? Experience
is measured in out-of-fold **patients**, so no patient is split across sizes,
and each size is a prefix of the next.

Two quantities per size:

  R(a) = E[dQ | a]   the mean internal reward of the actions the selector picks,
                     evaluated on the patients that size did not see
  endpoints          Dice and tumour-free-with-FP for the agent built on that
                     size (run separately, this script writes the models)

The fixed-strategy arm is the reference line: it uses no experience at all, so
it does not vary with experience size. Phase V section 10 made that arm
mandatory -- without it a rising curve looks like learning even when a selector
that never learned anything does better at every point.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import build, STRATEGIES
from q_model import patient_of

SIZES = [0.10, 0.25, 0.50, 1.00]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, nargs="+", required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--cfg", type=Path,
                    default=Path(r"D:\data\reports\phase5_action_value.json"))
    ap.add_argument("--outdir", type=Path, default=Path(r"D:\data\reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = build(args.table, args.cand, args.diagpkl)
    cfg = json.load(open(args.cfg, encoding="utf-8"))["cfg"]
    meta, y, X = d["meta"], d["y"], d["X"]
    groups = np.array([patient_of(m["case"]) for m in meta])

    # d_q is the internal reward; read it back in the same join order
    dq = []
    for tab in args.table:
        for r in csv.DictReader(open(tab, newline="", encoding="utf-8")):
            dq.append(float(r["d_q"]) if r["d_q"] not in ("", "nan") else np.nan)
    dq = np.array(dq, float)
    assert len(dq) == len(y), (len(dq), len(y))

    pats = sorted(set(groups))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pats)

    # Per-component candidate index, restricted to the components the agent
    # would actually be choosing among. The loop only ever reaches a component
    # whose diagnostic confidence clears the gate, so a curve computed over all
    # 34,999 actions would describe a selector operating on a population the
    # agent never sees -- and would make the fixed-strategy reference look far
    # worse than it is, since blanket removal is only sensible once the detector
    # is confident.
    GATE = 0.90
    idx = {}
    for i, m in enumerate(meta):
        if m["p_failure"] < GATE:
            continue
        idx.setdefault((m["case"], m["component_id"],
                        m["strategy"] == "recover"), []).append(i)
    print(f"gated at p_failure >= {GATE}: {len(idx)} components, "
          f"{sum(len(v) for v in idx.values())} candidate actions")

    rows = []
    for frac in SIZES:
        k = max(1, int(round(frac * len(pats))))
        train_pats = set(pats[:k])
        tr = np.array([g in train_pats for g in groups])
        te = ~tr
        if te.sum() == 0:                     # 100%: evaluate in-sample, flagged
            te = tr
            held_out = False
        else:
            held_out = True

        m = HistGradientBoostingRegressor(random_state=0, **cfg)
        m.fit(X[tr], y[tr])
        pred = m.predict(X)

        # the action this selector would take per component, among held-out ones
        sel = []
        for key, ix in idx.items():
            ix_te = [i for i in ix if te[i]]
            if ix_te:
                sel.append(max(ix_te, key=lambda i: pred[i]))
        sel = np.array(sel, int)
        R = float(np.nanmean(dq[sel]))
        dqd = float(np.nanmean(y[sel]))
        prec = float((y[sel] > 0).sum() /
                     max((y[sel] > 0).sum() + (y[sel] < 0).sum(), 1))

        out = args.outdir / f"fe_av_{int(frac*100):03d}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"model": m, "names": d["names"], "cfg": cfg,
                         "strategies": STRATEGIES,
                         "pred_classes": d["pred_classes"],
                         "feats": pickle.load(open(args.diagpkl, "rb"))
                         ["pred"]["feats"]}, fh)

        rows.append({"frac": frac, "patients": k, "actions_seen": int(tr.sum()),
                     "R_dq": R, "mean_d_qdice": dqd, "precision": prec,
                     "n_selected": len(sel), "held_out": held_out,
                     "model": str(out)})
        print(f"  {int(frac*100):3d}%  {k:3d} patients, {int(tr.sum()):6d} actions "
              f"| R(a)=E[dQ|a] {R:+.5f}  mean d_qdice {dqd:+.5f}  "
              f"precision {100*prec:.1f}%"
              f"{'' if held_out else '   (in-sample)'}")

    # the no-experience reference: always topology_remove
    fix = []
    for key, ix in idx.items():
        for i in ix:
            if meta[i]["strategy"] == "topology_remove":
                fix.append(i)
                break
    fix = np.array(fix, int)
    ref = {"R_dq": float(np.nanmean(dq[fix])),
           "mean_d_qdice": float(np.nanmean(y[fix])),
           "precision": float((y[fix] > 0).sum() /
                              max((y[fix] > 0).sum() + (y[fix] < 0).sum(), 1)),
           "n_selected": len(fix)}
    print(f"  fixed topology_remove (no experience): R(a) {ref['R_dq']:+.5f}  "
          f"mean d_qdice {ref['mean_d_qdice']:+.5f}  "
          f"precision {100*ref['precision']:.1f}%")

    p = args.outdir / "final_selfimprove.json"
    p.write_text(json.dumps({"curve": rows, "fixed_reference": ref}, indent=1),
                 encoding="utf-8")
    print(f"written {p}")


if __name__ == "__main__":
    main()
