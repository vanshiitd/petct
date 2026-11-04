#!/usr/bin/env python3
"""Phase XVI: the combined agent -- anchored union of the hand-built policy
and the guarded context detector.

See `reports/phase16_combined_plan.md`. Every hand-built action is kept with
its hand-built class (Phase XII anchor). The context detector, calibrated and
guarded as in Phase XIV and abstaining outside the Phase XV support fence,
may only *add* actions; an addition set is promoted only under Phase XII's
causal-utility criteria, else the fold rolls back to hand-built.

Writes, all new files:
  phase16_comb_{oof,val,test}.npz   --pfail-table (run with --min-pfail 1e-6)
  phase16_comb.json                 selection, promotion tables, descriptives
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, measured,
                                     labels_for, fit_stage1, predict_stage1,
                                     stage1_crossfit, write_table)
from phase12_anchored import stats, promotable
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import DETS, GATES, GUARDS, cal_fit, cal_apply, decide
from phase15_ood import case_fence, cal_crossfit_ind


def union(cfg, Po_raw, Po_cal, Pn_cal, solo_ok, ind):
    """(P, act, added): hand-built actions kept, in-support additions only."""
    hb_act = Po_raw.max(1) >= BASE_GATE
    if cfg is None:
        return Po_raw, hb_act, np.zeros(len(hb_act), bool)
    P, act = decide(cfg, Po_cal, Pn_cal, solo_ok)
    added = ind & ~hb_act & act
    return np.where(added[:, None], P, Po_raw), hb_act | added, added


def select(idx, Po_raw, Po_cal, Pn_cal, solo_ok, ind, keys, meas, policy):
    best, table = None, []
    for d in DETS:
        for t in GATES:
            for g in GUARDS:
                P, _, added = union((d, t, g), Po_raw, Po_cal, Pn_cal, solo_ok, ind)
                outs = labels_for(keys, P, meas, policy)
                ai = [i for i in idx if added[i] and outs[i] is not None]
                st = stats(ai, outs, keys)
                ok = promotable(st)
                table.append({"det": d, "t": t, "guard": g, **st, "promotable": ok})
                key = (st["sum"], d == "ens", t, GUARDS.index(g))
                if ok and (best is None or key > best[0]):
                    best = (key, (d, t, g))
    return (best[1] if best else None), table


def main() -> None:
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    old, new = features(), features() + NEW
    rows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    keys, Y = labels(rows, R / "diag_oof.pkl")
    fold = np.array([owner[r["case"]] for r in rows])
    X0, med0, _ = matrix(rows, old, None)
    X1, med1, _ = matrix(rows, new, None)
    meas = measured([R / "strategy_table_oof.csv", R / "strategy6_oof.csv"])
    ncase = collections.Counter(r["case"] for r in rows)
    n_row = np.array([ncase[r["case"]] for r in rows])
    solo_ok = n_row > 1

    P0 = stage1_crossfit(X0, Y, fold, list(FOLDS))
    P1 = stage1_crossfit(X1, Y, fold, list(FOLDS))
    for f, P in (("audit_crossfit.npz", P0), ("phase13_det_oof.npz", P1)):
        z = np.load(R / f, allow_pickle=True)
        ref = {(str(c), int(i)): p for c, i, p in
               zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
        dev = float(np.abs(P - np.array([ref[k] for k in keys])).max())
        print(f"route check: max |P - {f}| = {dev:.2e}")
        assert dev < 1e-9
    z = np.load(R / "audit_crossfit.npz", allow_pickle=True)
    rec_keys = [(str(c), int(i)) for c, i in zip(z["rec_keys_case"], z["rec_keys_cid"])]
    rec_cf = np.asarray(z["rec_crossfit"], float)

    report = {"plan": "reports/phase16_combined_plan.md", "outer": []}
    P_dep, act_dep = np.zeros_like(P0), np.zeros(len(rows), bool)
    for k in FOLDS:
        T = [f for f in FOLDS if f != k]
        tr, te = np.isin(fold, T), fold == k
        idx = list(np.where(tr)[0])
        ind = n_row <= case_fence(rows, idx)
        I0 = stage1_crossfit(X0, Y, fold, T)
        I1 = stage1_crossfit(X1, Y, fold, T)
        cfg, table = select(idx, I0, cal_crossfit_ind(I0, Y, fold, T, ind),
                            cal_crossfit_ind(I1, Y, fold, T, ind), solo_ok, ind,
                            keys, meas, policy)
        P0c = cal_apply(cal_fit(I0[tr & ind], Y[tr & ind]), P0[te])
        P1c = cal_apply(cal_fit(I1[tr & ind], Y[tr & ind]), P1[te])
        P, act, added = union(cfg, P0[te], P0c, P1c, solo_ok[te], ind[te])
        P_dep[te], act_dep[te] = P, act
        outs = labels_for([keys[i] for i in np.where(te)[0]], P, meas, policy)
        ai = [j for j in range(len(outs)) if added[j] and outs[j] is not None]
        held = stats(ai, outs, [keys[i] for i in np.where(te)[0]])
        n_prom = sum(r["promotable"] for r in table)
        print(f"outer fold {k}: {n_prom}/24 promotable -> {cfg or 'rollback'}   held-out "
              f"additions n={held['n']} sum={held['sum']:+.2f} lost={held['lost']:.0f} "
              f"harm={held['harm']}")
        report["outer"].append({"fold": k, "config": cfg, "train_table": table,
                                "heldout_added": held})
    write_table(R / "phase16_comb_oof.npz", keys, P_dep, act_dep, rec_keys, rec_cf)

    # ---- production ------------------------------------------------------
    allidx = list(range(len(rows)))
    fz = case_fence(rows, allidx)
    ind = n_row <= fz
    cfg, table = select(allidx, P0, cal_crossfit_ind(P0, Y, fold, list(FOLDS), ind),
                        cal_crossfit_ind(P1, Y, fold, list(FOLDS), ind), solo_ok, ind,
                        keys, meas, policy)
    best = max(table, key=lambda r: r["sum"] if r["promotable"] else -1e9)
    print(f"\nproduction: fence {fz:.1f}, {sum(r['promotable'] for r in table)}/24 "
          f"promotable -> {cfg or 'rollback'}  {best if best['promotable'] else ''}")
    report["production"] = {"fence": fz, "config": cfg, "table": table}
    C0, C1 = cal_fit(P0[ind], Y[ind]), cal_fit(P1[ind], Y[ind])
    M0, M1 = fit_stage1(X0, Y, np.arange(len(rows))), fit_stage1(X1, Y, np.arange(len(rows)))
    for split in ("val", "test"):
        rs, _, _, _ = load(R / f"cand_cv_{split}_ctx.csv", "pred")
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        Ps0 = predict_stage1(M0, matrix(rs, old, med0)[0])
        Ps1 = predict_stage1(M1, matrix(rs, new, med1)[0])
        nc = collections.Counter(r["case"] for r in rs)
        nr = np.array([nc[r["case"]] for r in rs])
        P, act, added = union(cfg, Ps0, cal_apply(C0, Ps0), cal_apply(C1, Ps1), nr > 1,
                              nr <= fz)
        hbz = np.load(R / f"phase11_eag_hb_{split}.npz", allow_pickle=True)
        write_table(R / f"phase16_comb_{split}.npz", ks, P, act,
                    [(str(c), int(i)) for c, i in zip(hbz["rec_keys_case"], hbz["rec_keys_cid"])],
                    np.asarray(hbz["rec_crossfit"], float))
        # counts only -- no outcome is read for test
        report[f"production_{split}"] = {"acted": int(act.sum()), "added": int(added.sum()),
                                         "handbuilt_gated": int((Ps0.max(1) >= BASE_GATE).sum())}
        print(f"  {split}: {report[f'production_{split}']}")
    (R / "phase16_comb.json").write_text(json.dumps(report, indent=1, default=float),
                                         encoding="utf-8")
    print(f"written {R / 'phase16_comb.json'}")


if __name__ == "__main__":
    main()
