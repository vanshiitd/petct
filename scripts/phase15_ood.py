#!/usr/bin/env python3
"""Phase XV: Phase XIV plus out-of-distribution abstention.

See `reports/phase15_ood_abstention_plan.md`. The only change from
`phase14_calibrated.py`: a case whose number of predicted components lies
above the training Q3 + 3 IQR fence is handed to the hand-built detector
(old, uncalibrated, 0.90 gate). The same fence keeps such cases out of
calibrator fitting and scores them with the hand-built decision in the
admissibility statistics.

Writes, all new files:
  phase15_ood_{oof,val,test}.npz   --pfail-table (run with --min-pfail 1e-6)
  phase15_ood.json                 fences, selection, descriptives
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
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import (DETS, GATES, GUARDS, cal_fit, cal_apply, decide, score,
                                overconf)


def fence(counts) -> float:
    q1, q3 = np.percentile(np.asarray(counts, float), [25, 75])
    return float(q3 + 3.0 * (q3 - q1))


def case_fence(rows, idx) -> float:
    c = collections.Counter(rows[i]["case"] for i in idx)
    return fence(list(c.values()))


def cal_crossfit_ind(P, Y, fold, within, ind):
    out = np.full_like(P, np.nan)
    for j in within:
        tr = np.isin(fold, [k for k in within if k != j]) & ind
        out[fold == j] = cal_apply(cal_fit(P[tr], Y[tr]), P[fold == j])
    return out


def deploy(cfg, Po_raw, Po_cal, Pn_cal, solo_ok, ind):
    """The configuration in support, the hand-built decision outside it."""
    hb_act = Po_raw.max(1) >= BASE_GATE
    if cfg is None:
        return Po_raw, hb_act
    P, act = decide(cfg, Po_cal, Pn_cal, solo_ok)
    P = np.where(ind[:, None], P, Po_raw)
    return P, np.where(ind, act, hb_act)


def select(idx, Po_raw, Po_cal, Pn_cal, solo_ok, ind, keys, meas, policy):
    hb = score(idx, Po_raw.max(1) >= BASE_GATE, labels_for(keys, Po_raw, meas, policy))
    best, table = None, []
    for d in DETS:
        for t in GATES:
            for g in GUARDS:
                P, act = deploy((d, t, g), Po_raw, Po_cal, Pn_cal, solo_ok, ind)
                st = score(idx, act, labels_for(keys, P, meas, policy))
                ok = st["lost"] <= hb["lost"] and st["pos_sum"] >= hb["pos_sum"]
                table.append({"det": d, "t": t, "guard": g, **st, "admissible": ok})
                key = (st["sum"], d == "ens", t, GUARDS.index(g))
                if ok and (best is None or key > best[0]):
                    best = (key, (d, t, g))
    return (best[1] if best else None), hb, table


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
    tp = np.array([r["status"] == "TP" for r in rows])

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

    report = {"plan": "reports/phase15_ood_abstention_plan.md", "outer": []}
    P_dep, act_dep = np.zeros_like(P0), np.zeros(len(rows), bool)
    ind_oof = np.zeros(len(rows), bool)
    P0c_out, P1c_out = np.full_like(P0, np.nan), np.full_like(P1, np.nan)
    for k in FOLDS:
        T = [f for f in FOLDS if f != k]
        tr, te = np.isin(fold, T), fold == k
        idx = list(np.where(tr)[0])
        fz = case_fence(rows, idx)
        ind = n_row <= fz
        I0 = stage1_crossfit(X0, Y, fold, T)
        I1 = stage1_crossfit(X1, Y, fold, T)
        cfg, hb, table = select(idx, I0, cal_crossfit_ind(I0, Y, fold, T, ind),
                                cal_crossfit_ind(I1, Y, fold, T, ind), solo_ok, ind,
                                keys, meas, policy)
        P0c_out[te] = cal_apply(cal_fit(I0[tr & ind], Y[tr & ind]), P0[te])
        P1c_out[te] = cal_apply(cal_fit(I1[tr & ind], Y[tr & ind]), P1[te])
        P_dep[te], act_dep[te] = deploy(cfg, P0[te], P0c_out[te], P1c_out[te],
                                        solo_ok[te], ind[te])
        ind_oof[te] = ind[te]
        held = score(list(np.where(te)[0]), act_dep, labels_for(keys, P_dep, meas, policy))
        held_hb = score(list(np.where(te)[0]), P0.max(1) >= BASE_GATE,
                        labels_for(keys, P0, meas, policy))
        out_tr = sorted({rows[i]["case"] for i in idx if not ind[i]})
        out_te = sorted({rows[i]["case"] for i in np.where(te & ~ind)[0]})
        n_adm = sum(r["admissible"] for r in table)
        print(f"outer fold {k}: fence {fz:.1f} | abstain train {len(out_tr)} cases, held-out "
              f"{len(out_te)} {out_te} | {n_adm}/24 admissible -> {cfg or 'rollback'}")
        print(f"   held-out: chosen n={held['n']} sum={held['sum']:+.2f} lost={held['lost']:.0f}"
              f"  | hand-built n={held_hb['n']} sum={held_hb['sum']:+.2f} lost={held_hb['lost']:.0f}")
        report["outer"].append({"fold": k, "fence": fz, "abstain_train": out_tr,
                                "abstain_heldout": out_te, "config": cfg,
                                "train_handbuilt": hb, "train_table": table,
                                "heldout": held, "heldout_handbuilt": held_hb})
    write_table(R / "phase15_ood_oof.npz", keys, P_dep, act_dep, rec_keys, rec_cf)

    m = ind_oof
    report["overconfidence_oof_in_support"] = {
        n: overconf(P[m], Y[m], tp[m]) for n, P in
        (("old_raw", P0), ("new_raw", P1), ("new_cal", P1c_out),
         ("ens", (P0c_out + P1c_out) / 2.0))}

    # ---- production ------------------------------------------------------
    allidx = list(range(len(rows)))
    fz = case_fence(rows, allidx)
    ind = n_row <= fz
    cfg, hb, table = select(allidx, P0, cal_crossfit_ind(P0, Y, fold, list(FOLDS), ind),
                            cal_crossfit_ind(P1, Y, fold, list(FOLDS), ind), solo_ok, ind,
                            keys, meas, policy)
    print(f"\nproduction: fence {fz:.1f}, {sum(r['admissible'] for r in table)}/24 "
          f"admissible -> {cfg or 'rollback'}")
    report["production"] = {"fence": fz, "config": cfg, "handbuilt": hb, "table": table}
    C0, C1 = cal_fit(P0[ind], Y[ind]), cal_fit(P1[ind], Y[ind])
    M0, M1 = fit_stage1(X0, Y, np.arange(len(rows))), fit_stage1(X1, Y, np.arange(len(rows)))
    for split in ("val", "test"):
        rs, _, _, _ = load(R / f"cand_cv_{split}_ctx.csv", "pred")
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        Ps0 = predict_stage1(M0, matrix(rs, old, med0)[0])
        Ps1 = predict_stage1(M1, matrix(rs, new, med1)[0])
        nc = collections.Counter(r["case"] for r in rs)
        nr = np.array([nc[r["case"]] for r in rs])
        P, act = deploy(cfg, Ps0, cal_apply(C0, Ps0), cal_apply(C1, Ps1), nr > 1, nr <= fz)
        hbz = np.load(R / f"phase11_eag_hb_{split}.npz", allow_pickle=True)
        write_table(R / f"phase15_ood_{split}.npz", ks, P, act,
                    [(str(c), int(i)) for c, i in zip(hbz["rec_keys_case"], hbz["rec_keys_cid"])],
                    np.asarray(hbz["rec_crossfit"], float))
        # counts only -- no outcome is read for test
        report[f"production_{split}"] = {
            "acted": int(act.sum()), "handbuilt_gated": int((Ps0.max(1) >= BASE_GATE).sum()),
            "abstain_cases": sorted({r["case"] for r, n in zip(rs, nr) if n > fz})}
        print(f"  {split}: abstaining cases {report[f'production_{split}']['abstain_cases']}")

    print("\nin-support overconfidence (out-of-fold):")
    for n, c in report["overconfidence_oof_in_support"].items():
        print(f"  {n:8s} ECE {c['ece']:.3f}  at gate {c['n_at_gate']:5d}  "
              f"precision@gate {c['precision_at_gate']:.3f}  "
              f"true lesion -> spurious@gate {c['true_lesion_called_spurious_at_gate']:.4f}")
    (R / "phase15_ood.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print(f"written {R / 'phase15_ood.json'}")


if __name__ == "__main__":
    main()
