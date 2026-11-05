#!/usr/bin/env python3
"""Phase XVII: Learn-then-Test certification of the Phase XVI additions.

See `reports/phase17_ltt_plan.md`. Each candidate update (detector x guard x
gate, as Phase XVI's anchored union) is certified only if the exact binomial
test rejects H: P(patient harmed by the update) > DELTA, with fixed-sequence
testing over gates inside each detector x guard chain and Bonferroni across
chains. Certified candidates must also pass Phase XII's n and benefit-LCB
checks; else roll back to the hand-built policy.

Writes, all new files:
  phase17_ltt_{oof,val,test}.npz   --pfail-table (run with --min-pfail 1e-6)
  phase17_ltt.json                 candidates, p-values, curve, production
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, patient_of
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, measured,
                                     labels_for, fit_stage1, predict_stage1,
                                     stage1_crossfit, write_table)
from phase12_anchored import stats, MIN_N
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import DETS, GUARDS, cal_fit, cal_apply
from phase15_ood import case_fence, cal_crossfit_ind
from phase16_combined import union

DELTA, ALPHA = 0.01, 0.05
CHAIN_GATES = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85]
CURVE_DELTA = [0.005, 0.01, 0.02, 0.03, 0.05]
CURVE_ALPHA = [0.05, 0.10]


def harmed_patients(added, outs, keys) -> set:
    return {patient_of(keys[i][0]) for i in np.where(added)[0]
            if outs[i] is not None and outs[i]["harm"]}


def candidates(Po_raw, Po_cal, Pn_cal, solo_ok, ind, keys, meas, policy):
    """Every (det, guard, gate) with its harmed-patient count and benefit stats."""
    out = {}
    for d in DETS:
        for g in GUARDS:
            for t in CHAIN_GATES:
                P, _, added = union((d, t, g), Po_raw, Po_cal, Pn_cal, solo_ok, ind)
                outs = labels_for(keys, P, meas, policy)
                ai = [i for i in np.where(added)[0] if outs[i] is not None]
                out[(d, g, t)] = {"k": len(harmed_patients(added, outs, keys)),
                                  **stats(ai, outs, keys)}
    return out


def certify(cands, n_pat, delta, alpha):
    """LTT: fixed sequence over gates per chain at alpha / n_chains."""
    chains = [(d, g) for d in DETS for g in GUARDS]
    level = alpha / len(chains)
    cert = []
    for d, g in chains:
        for t in CHAIN_GATES:
            if binom.cdf(cands[(d, g, t)]["k"], n_pat, delta) > level:
                break
            cert.append((d, g, t))
    return cert


def choose(cands, cert):
    best = None
    for d, g, t in cert:
        c = cands[(d, g, t)]
        if c["n"] < MIN_N or not c["lcb"] > 0:
            continue
        key = (c["sum"], t, d == "ens", GUARDS.index(g))
        if best is None or key > best[0]:
            best = (key, (d, t, g))
    return best[1] if best else None


def main() -> None:
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    n_pat = len({patient_of(s) for s in owner})
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

    fz = case_fence(rows, list(range(len(rows))))
    ind = n_row <= fz
    P0c = cal_crossfit_ind(P0, Y, fold, list(FOLDS), ind)
    P1c = cal_crossfit_ind(P1, Y, fold, list(FOLDS), ind)
    cands = candidates(P0, P0c, P1c, solo_ok, ind, keys, meas, policy)
    k_min = int(binom.ppf(ALPHA / 8, n_pat, DELTA))
    while k_min >= 0 and binom.cdf(k_min, n_pat, DELTA) > ALPHA / 8:
        k_min -= 1
    print(f"patients {n_pat}, fence {fz:.1f}; largest certifiable k at delta={DELTA}, "
          f"alpha/8={ALPHA / 8:.4f}: {k_min}")

    cert = certify(cands, n_pat, DELTA, ALPHA)
    cfg = choose(cands, cert)
    print(f"\ncertified {len(cert)}/{len(cands)}; deployed: {cfg or 'rollback'}")
    for d in DETS:
        for g in GUARDS:
            line = " ".join(f"{t:.2f}:k{cands[(d, g, t)]['k']}"
                            f"{'*' if (d, g, t) in cert else ''}" for t in CHAIN_GATES)
            print(f"  {d:7s} {g:10s} {line}")
    if cfg:
        c = cands[(cfg[0], cfg[2], cfg[1])]
        print(f"  chosen: n={c['n']} sum={c['sum']:+.2f} lcb={c['lcb']:+.4f} "
              f"k={c['k']} lost={c['lost']:.0f} pos_sum={c['pos_sum']:+.2f}")

    curve = []
    for a in CURVE_ALPHA:
        for dl in CURVE_DELTA:
            ct = certify(cands, n_pat, dl, a)
            cf = choose(cands, ct)
            c = cands[(cf[0], cf[2], cf[1])] if cf else None
            curve.append({"alpha": a, "delta": dl, "n_certified": len(ct), "config": cf,
                          "n_added": c["n"] if c else 0, "sum": c["sum"] if c else 0.0,
                          "k": c["k"] if c else 0, "lost": c["lost"] if c else 0.0})
            print(f"  curve alpha={a:.2f} delta={dl:.3f}: certified {len(ct):2d} -> "
                  f"{cf or 'rollback'} sum={curve[-1]['sum']:+.2f} k={curve[-1]['k']}")

    report = {"plan": "reports/phase17_ltt_plan.md", "delta": DELTA, "alpha": ALPHA,
              "n_patients": n_pat, "fence": fz, "k_max_certifiable": k_min,
              "candidates": [{"det": d, "guard": g, "t": t, **c,
                              "p": float(binom.cdf(c["k"], n_pat, DELTA)),
                              "certified": (d, g, t) in cert}
                             for (d, g, t), c in cands.items()],
              "config": cfg, "curve": curve}

    P_oof, act_oof, _ = union(cfg, P0, P0c, P1c, solo_ok, ind)
    write_table(R / "phase17_ltt_oof.npz", keys, P_oof, act_oof, rec_keys, rec_cf)
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
        write_table(R / f"phase17_ltt_{split}.npz", ks, P, act,
                    [(str(c), int(i)) for c, i in zip(hbz["rec_keys_case"], hbz["rec_keys_cid"])],
                    np.asarray(hbz["rec_crossfit"], float))
        # counts only -- no outcome is read for test
        report[f"production_{split}"] = {"acted": int(act.sum()), "added": int(added.sum()),
                                         "handbuilt_gated": int((Ps0.max(1) >= BASE_GATE).sum())}
        print(f"  {split}: {report[f'production_{split}']}")
    (R / "phase17_ltt.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print(f"written {R / 'phase17_ltt.json'}")


if __name__ == "__main__":
    main()
