#!/usr/bin/env python3
"""Phase XVII-b: lifelong two-round certification and the LTT validity check.

See `reports/phase17b_lifelong_validity_plan.md`. Descriptive only: nothing
here is deployed and no test outcome is read.

Writes: phase17b_lifelong.json
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, patient_of
from phase11_experience_gate import (R, SPLITS, FOLDS, features, measured, labels_for,
                                     fit_stage1, predict_stage1, stage1_crossfit)
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import DETS, GUARDS, cal_fit, cal_apply
from phase15_ood import case_fence, cal_crossfit_ind
from phase16_combined import union
from phase17_ltt import CHAIN_GATES, ALPHA, candidates, certify, choose

ROUNDS = [((3, 4), (5, 6, 7)), ((3, 4, 5), (6, 7))]
DELTAS = [0.02, 0.03, 0.05]
N_SPLIT, SEED = 1000, 2026
CHAINS = [(d, g) for d in DETS for g in GUARDS]


def k0_n(alpha, delta):
    return math.ceil(math.log(alpha / len(CHAINS)) / math.log(1 - delta))


def per_patient(cfg, Po_raw, Po_cal, Pn_cal, solo_ok, ind, keys, meas, policy, pid):
    """Patient harm indicator and summed d_qdice of the additions."""
    P, _, added = union(cfg, Po_raw, Po_cal, Pn_cal, solo_ok, ind)
    outs = labels_for(keys, P, meas, policy)
    h, b = collections.defaultdict(int), collections.defaultdict(float)
    for i in np.where(added)[0]:
        if outs[i] is not None:
            p = patient_of(keys[i][0])
            h[p] |= int(outs[i]["harm"])
            b[p] += outs[i]["y"]
    return np.array([h[p] for p in pid]), np.array([b[p] for p in pid])


def ltt_fast(H, B, delta, alpha):
    """Certified candidate index with largest summed benefit, or None; H, B: cand x patients."""
    n = H.shape[1]
    k = H.sum(1)
    best, j = None, 0
    for _ in CHAINS:
        for _ in CHAIN_GATES:
            if binom.cdf(k[j], n, delta) > alpha / len(CHAINS):
                j += len(CHAIN_GATES) - (j % len(CHAIN_GATES))
                break
            if best is None or B[j].sum() > B[best].sum():
                best = j
            j += 1
    return best


def main() -> None:
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    old, new = features(), features() + NEW
    rows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    keys, Y = labels(rows, R / "diag_oof.pkl")
    fold = np.array([owner[r["case"]] for r in rows])
    X0, _, _ = matrix(rows, old, None)
    X1, _, _ = matrix(rows, new, None)
    meas = measured([R / "strategy_table_oof.csv", R / "strategy6_oof.csv"])
    ncase = collections.Counter(r["case"] for r in rows)
    n_row = np.array([ncase[r["case"]] for r in rows])
    solo_ok = n_row > 1
    report = {"plan": "reports/phase17b_lifelong_validity_plan.md", "rounds": [],
              "validity": []}

    # ---- lifelong rounds -------------------------------------------------
    for r, (T, C) in enumerate(ROUNDS, 1):
        a_r = ALPHA * 2.0 ** -r
        tr, ce = np.isin(fold, T), np.isin(fold, C)
        idx = np.where(tr)[0]
        ind = n_row <= case_fence(rows, list(idx))
        I0, I1 = stage1_crossfit(X0, Y, fold, list(T)), stage1_crossfit(X1, Y, fold, list(T))
        C0, C1 = cal_fit(I0[tr & ind], Y[tr & ind]), cal_fit(I1[tr & ind], Y[tr & ind])
        Q0 = predict_stage1(fit_stage1(X0, Y, idx), X0[ce])
        Q1 = predict_stage1(fit_stage1(X1, Y, idx), X1[ce])
        kc = [keys[i] for i in np.where(ce)[0]]
        n_pat = len({patient_of(s) for s in owner if owner[s] in C})
        cands = candidates(Q0, cal_apply(C0, Q0), cal_apply(C1, Q1), solo_ok[ce], ind[ce],
                           kc, meas, policy)
        for dl in DELTAS:
            cert = certify(cands, n_pat, dl, a_r)
            cfg = choose(cands, cert)
            c = cands[(cfg[0], cfg[2], cfg[1])] if cfg else None
            row = {"round": r, "train": T, "certify": C, "alpha_r": a_r, "delta": dl,
                   "n_patients": n_pat, "k0_n_required": k0_n(a_r, dl),
                   "n_certified": len(cert), "config": cfg,
                   **({k: c[k] for k in ("n", "sum", "k", "lost", "pos_sum", "lcb")} if c
                      else {"n": 0, "sum": 0.0, "k": 0, "lost": 0.0})}
            report["rounds"].append(row)
            print(f"round {r} delta={dl:.2f} a_r={a_r:.4f} n_pat={n_pat} "
                  f"(k=0 needs {row['k0_n_required']}): certified {len(cert):2d} -> "
                  f"{cfg or 'rollback'} n={row['n']} sum={row['sum']:+.2f} k={row['k']} "
                  f"lost={row['lost']:.0f}")

    # ---- validity check --------------------------------------------------
    P0 = stage1_crossfit(X0, Y, fold, list(FOLDS))
    P1 = stage1_crossfit(X1, Y, fold, list(FOLDS))
    ind = n_row <= case_fence(rows, list(range(len(rows))))
    P0c = cal_crossfit_ind(P0, Y, fold, list(FOLDS), ind)
    P1c = cal_crossfit_ind(P1, Y, fold, list(FOLDS), ind)
    pid = sorted({patient_of(s) for s in owner})
    HB = [per_patient((d, t, g), P0, P0c, P1c, solo_ok, ind, keys, meas, policy, pid)
          for d, g in CHAINS for t in CHAIN_GATES]
    H = np.array([h for h, _ in HB])
    B = np.array([b for _, b in HB])
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(len(pid)) for _ in range(N_SPLIT)]
    for dl in DELTAS:
        res = {"cert": [], "naive": []}
        for perm in splits:
            ca, ev = perm[: len(pid) // 2], perm[len(pid) // 2:]
            j = ltt_fast(H[:, ca], B[:, ca], dl, ALPHA)
            jn = int(np.argmax(B[:, ca].sum(1)))
            res["cert"].append((H[j, ev].mean(), B[j, ev].sum()) if j is not None
                               else (0.0, 0.0, "rollback"))
            res["naive"].append((H[jn, ev].mean(), B[jn, ev].sum()))
        out = {"delta": dl, "n_cal": len(pid) // 2, "k0_n_required": k0_n(ALPHA, dl)}
        for name, v in res.items():
            risk = np.array([x[0] for x in v])
            ben = np.array([x[1] for x in v])
            out[name] = {"violation_freq": float((risk > dl).mean()),
                         "mean_risk": float(risk.mean()),
                         "p95_risk": float(np.percentile(risk, 95)),
                         "mean_benefit": float(ben.mean()),
                         "rollback_freq": float(np.mean([len(x) == 3 for x in v]))}
        report["validity"].append(out)
        print(f"validity delta={dl:.2f}: certified viol={out['cert']['violation_freq']:.3f} "
              f"risk={out['cert']['mean_risk']:.4f} ben={out['cert']['mean_benefit']:+.2f} "
              f"rollback={out['cert']['rollback_freq']:.2f} | naive "
              f"viol={out['naive']['violation_freq']:.3f} risk={out['naive']['mean_risk']:.4f} "
              f"ben={out['naive']['mean_benefit']:+.2f}")
    (R / "phase17b_lifelong.json").write_text(json.dumps(report, indent=1, default=float),
                                              encoding="utf-8")
    print(f"written {R / 'phase17b_lifelong.json'}")


if __name__ == "__main__":
    main()
