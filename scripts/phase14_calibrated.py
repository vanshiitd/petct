#!/usr/bin/env python3
"""Phase XIV: calibrated, safety-guarded context detector.

See `reports/phase14_calibrated_detector_plan.md`. Four defences against the
Phase XIII detector's overconfidence -- nested isotonic calibration, an
ensemble with the old detector, an agreement guard and a solo-component
guard -- plus the gate, 24 configurations in all. On each outer fold a
configuration is admissible only if, on the training folds, it loses no more
lesions and no more tumour-positive d_qdice than the hand-built reference;
the best admissible one is taken, else the fold rolls back to hand-built.

Writes, all new files:
  phase14_cal_{oof,val,test}.npz   --pfail-table (run with --min-pfail 1e-6)
  phase14_cal.json                 selection, overconfidence descriptives
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase4_diag_model import PRED_CLASSES
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, measured,
                                     labels_for, fit_stage1, predict_stage1,
                                     stage1_crossfit, calibration, write_table)
from phase13_context_features import NEW
from phase13_detector import labels

DETS = ["new_cal", "ens"]
GATES = [0.85, 0.90, 0.95]
GUARDS = ["none", "solo", "agree", "agree+solo"]     # weakest to strongest
AGREE_FLOOR = 0.50
SPURIOUS = [PRED_CLASSES.index(c) for c in
            ("spurious_physio", "spurious_liver_spleen", "spurious_other")]


# --------------------------------------------------------- calibration ----
def cal_fit(P, Y):
    cals = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(P[:, j])
        if Y[ok, j].sum() < 10 or np.all(P[ok, j] == 0):
            cals.append(None)
        else:
            cals.append(IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                        .fit(P[ok, j], Y[ok, j]))
    return cals


def cal_apply(cals, P):
    return np.column_stack([c.predict(P[:, j]) if c is not None else P[:, j]
                            for j, c in enumerate(cals)])


def cal_crossfit(P, Y, fold, within):
    """Each fold in `within` calibrated by calibrators fitted on the others."""
    out = np.full_like(P, np.nan)
    for j in within:
        tr = np.isin(fold, [k for k in within if k != j])
        out[fold == j] = cal_apply(cal_fit(P[tr], Y[tr]), P[fold == j])
    return out


# ------------------------------------------------------------- configs ----
def decide(cfg, Po_cal, Pn_cal, solo_ok):
    det, t, guard = cfg
    P = Pn_cal if det == "new_cal" else (Po_cal + Pn_cal) / 2.0
    act = P.max(1) >= t
    if "agree" in guard:
        act &= (Po_cal.argmax(1) == Pn_cal.argmax(1)) & (Po_cal.max(1) >= AGREE_FLOOR)
    if "solo" in guard:
        act &= solo_ok
    return P, act


def score(idx, act, outs):
    s = {"n": 0, "sum": 0.0, "lost": 0.0, "pos_sum": 0.0, "neg_sum": 0.0}
    for i in idx:
        if act[i] and outs[i] is not None:
            o = outs[i]
            s["n"] += 1
            s["sum"] += o["y"]
            s["lost"] += o["lost"]
            s["pos_sum" if o["pos"] else "neg_sum"] += o["y"]
    return s


def select(idx, Po_raw, Po_cal, Pn_cal, solo_ok, keys, meas, policy):
    hb_act = Po_raw.max(1) >= BASE_GATE
    hb = score(idx, hb_act, labels_for(keys, Po_raw, meas, policy))
    best, table = None, []
    for d in DETS:
        for t in GATES:
            for g in GUARDS:
                P, act = decide((d, t, g), Po_cal, Pn_cal, solo_ok)
                st = score(idx, act, labels_for(keys, P, meas, policy))
                ok = st["lost"] <= hb["lost"] and st["pos_sum"] >= hb["pos_sum"]
                table.append({"det": d, "t": t, "guard": g, **st, "admissible": ok})
                key = (st["sum"], d == "ens", t, GUARDS.index(g))
                if ok and (best is None or key > best[0]):
                    best = (key, (d, t, g))
    return (best[1] if best else None), hb, table


# ------------------------------------------------------- overconfidence ----
def overconf(P, Y, tp):
    pm, am = P.max(1), P.argmax(1)
    right = Y[np.arange(len(Y)), am]
    c = calibration(pm, right)
    c.pop("reliability")
    g = pm >= BASE_GATE
    c["n_at_gate"] = int(g.sum())
    c["precision_at_gate"] = float(right[g].mean()) if g.any() else float("nan")
    c["true_lesion_called_spurious_at_gate"] = float(
        (tp & g & np.isin(am, SPURIOUS)).sum() / max(1, tp.sum()))
    return c


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
    solo_ok = np.array([ncase[r["case"]] > 1 for r in rows])
    tp = np.array([r["status"] == "TP" for r in rows])

    # outer-fold raw predictions; checked against the deployed tables
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

    report = {"plan": "reports/phase14_calibrated_detector_plan.md", "outer": []}
    P_dep = np.zeros_like(P0)
    act_dep = np.zeros(len(rows), bool)
    P0c_out, P1c_out = np.full_like(P0, np.nan), np.full_like(P1, np.nan)
    for k in FOLDS:
        T = [f for f in FOLDS if f != k]
        tr, te = np.isin(fold, T), fold == k
        idx = list(np.where(tr)[0])
        I0 = stage1_crossfit(X0, Y, fold, T)
        I1 = stage1_crossfit(X1, Y, fold, T)
        cfg, hb, table = select(idx, I0, cal_crossfit(I0, Y, fold, T),
                                cal_crossfit(I1, Y, fold, T), solo_ok, keys, meas, policy)
        P0c_out[te] = cal_apply(cal_fit(I0[tr], Y[tr]), P0[te])
        P1c_out[te] = cal_apply(cal_fit(I1[tr], Y[tr]), P1[te])
        if cfg is None:
            P_dep[te], act_dep[te] = P0[te], P0[te].max(1) >= BASE_GATE
        else:
            P, act = decide(cfg, P0c_out[te], P1c_out[te], solo_ok[te])
            P_dep[te], act_dep[te] = P, act
        held = score(list(np.where(te)[0]), act_dep, labels_for(keys, P_dep, meas, policy))
        held_hb = score(list(np.where(te)[0]), P0.max(1) >= BASE_GATE,
                        labels_for(keys, P0, meas, policy))
        n_adm = sum(r["admissible"] for r in table)
        print(f"outer fold {k}: {n_adm}/24 admissible -> {cfg or 'rollback'}   "
              f"train hand-built lost={hb['lost']:.0f} pos={hb['pos_sum']:+.2f}")
        print(f"   held-out: chosen n={held['n']} sum={held['sum']:+.2f} lost={held['lost']:.0f}"
              f"  | hand-built n={held_hb['n']} sum={held_hb['sum']:+.2f} lost={held_hb['lost']:.0f}")
        report["outer"].append({"fold": k, "config": cfg, "train_handbuilt": hb,
                                "train_table": table, "heldout": held,
                                "heldout_handbuilt": held_hb})
    write_table(R / "phase14_cal_oof.npz", keys, P_dep, act_dep, rec_keys, rec_cf)

    ens_out = (P0c_out + P1c_out) / 2.0
    report["overconfidence_oof"] = {n: overconf(P, Y, tp) for n, P in
                                    (("old_raw", P0), ("new_raw", P1),
                                     ("new_cal", P1c_out), ("ens", ens_out))}

    # ---- production ------------------------------------------------------
    allidx = list(range(len(rows)))
    cfg, hb, table = select(allidx, P0, cal_crossfit(P0, Y, fold, list(FOLDS)),
                            cal_crossfit(P1, Y, fold, list(FOLDS)), solo_ok, keys, meas, policy)
    print(f"\nproduction: {cfg or 'rollback'}")
    report["production"] = {"config": cfg, "handbuilt": hb, "table": table}
    C0, C1 = cal_fit(P0, Y), cal_fit(P1, Y)
    M0, M1 = fit_stage1(X0, Y, np.arange(len(rows))), fit_stage1(X1, Y, np.arange(len(rows)))
    for split in ("val", "test"):
        rs, _, _, _ = load(R / f"cand_cv_{split}_ctx.csv", "pred")
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        Ps0 = predict_stage1(M0, matrix(rs, old, med0)[0])
        Ps1 = predict_stage1(M1, matrix(rs, new, med1)[0])
        nc = collections.Counter(r["case"] for r in rs)
        so = np.array([nc[r["case"]] > 1 for r in rs])
        if cfg is None:
            P, act = Ps0, Ps0.max(1) >= BASE_GATE
        else:
            P, act = decide(cfg, cal_apply(C0, Ps0), cal_apply(C1, Ps1), so)
        hbz = np.load(R / f"phase11_eag_hb_{split}.npz", allow_pickle=True)
        write_table(R / f"phase14_cal_{split}.npz", ks, P, act,
                    [(str(c), int(i)) for c, i in zip(hbz["rec_keys_case"], hbz["rec_keys_cid"])],
                    np.asarray(hbz["rec_crossfit"], float))
        report[f"production_{split}"] = {"acted": int(act.sum()),
                                         "handbuilt_gated": int((Ps0.max(1) >= BASE_GATE).sum())}
        if split == "val":
            _, Yv = labels(rs, R / "diag_val.pkl")
            tpv = np.array([r["status"] == "TP" for r in rs])
            report["overconfidence_val"] = {
                "old_raw": overconf(Ps0, Yv, tpv), "new_raw": overconf(Ps1, Yv, tpv),
                "new_cal": overconf(cal_apply(C1, Ps1), Yv, tpv),
                "ens": overconf((cal_apply(C0, Ps0) + cal_apply(C1, Ps1)) / 2, Yv, tpv)}

    for part in ("overconfidence_oof", "overconfidence_val"):
        print(f"\n{part}:")
        for n, c in report[part].items():
            print(f"  {n:8s} ECE {c['ece']:.3f}  Brier {c['brier']:.3f}  at gate {c['n_at_gate']:5d}"
                  f"  precision@gate {c['precision_at_gate']:.3f}  "
                  f"true lesion -> spurious@gate {c['true_lesion_called_spurious_at_gate']:.4f}")
    (R / "phase14_cal.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print(f"written {R / 'phase14_cal.json'}")


if __name__ == "__main__":
    main()
