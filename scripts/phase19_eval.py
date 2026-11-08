#!/usr/bin/env python3
"""Phase XIX, after the lock: safety on C per arm, the risk-benefit curve and
the LTT validity check on B u C.

See `reports/phase19_psma_selfimprove_plan.md` sections 5 and 7. Dice
superiority (B19) and the NI endpoints come from
`phase13_eval.py --split psma --run p19_{T}_p19 --hb p19_{T}_hb`.

Writes phase19_eval_{T}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import patient_of
from phase5_eval import read
from phase11_experience_gate import R, measured
from phase17_ltt import CHAIN_GATES, candidates, certify, choose
from phase17b_lifelong import CHAINS, per_patient, ltt_fast
from phase19_selfimprove import DELTA, ALPHA, SPLIT, naive

CURVE = [0.03, 0.05, 0.10, 0.15, 0.20]
VALID = [0.05, 0.10]
N_SPLIT, SEED = 1000, 2027


def ub(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))


def safety(T, arms):
    hb = read(R / f"p19_{T}_hb", "final")
    base = read(R / f"p19_{T}_hb", "base")
    out = {}
    for a in arms:
        x = read(R / f"p19_{T}_{a}", "final")
        assert set(x) == set(hb)
        cs = sorted(x)
        harmed = [c for c in cs if x[c]["n_hit"] < hb[c]["n_hit"]
                  or (base[c]["n_gt"] > 0 and x[c]["dice"] - hb[c]["dice"] < -0.01)]
        n, k = len(cs), len(harmed)
        pos = [c for c in cs if base[c]["n_gt"] > 0]
        out[a] = {"n": n, "harmed": k, "p": float(binom.cdf(k, n, DELTA)), "ub95": ub(k, n),
                  "safe": bool(binom.cdf(k, n, DELTA) < ALPHA),
                  "dDice_pos": float(np.mean([x[c]["dice"] - hb[c]["dice"] for c in pos])),
                  "dFP_ml": float(np.mean([x[c]["fp_ml"] - hb[c]["fp_ml"] for c in cs])),
                  "d_hits": int(sum(x[c]["n_hit"] - hb[c]["n_hit"] for c in cs)),
                  "changed": sum(any(x[c][m] != hb[c][m] for m in ("dice", "fp_ml", "n_hit"))
                                 for c in cs),
                  "harmed_cases": harmed}
        o = out[a]
        print(f"  {a:7s} harmed {k}/{n} p(H0: R>{DELTA})={o['p']:.4f} UB {o['ub95']:.4f} | "
              f"dDice(pos) {o['dDice_pos']:+.4f} dFP {o['dFP_ml']:+.2f} mL d_hits {o['d_hits']:+d}"
              f" | changed {o['changed']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", default=["p19", "naive", "pseudo", "uncal"])
    args = ap.parse_args()
    T = args.tag
    sp = json.loads(SPLIT.read_text())
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    rep = {"tag": T}

    print(f"[{T}] safety on C vs hand-built")
    rep["safety"] = safety(T, args.arms)

    z = np.load(R / f"phase19_{T}_probs.npz")
    meas = measured([R / f"strategy_table_{T}.csv"])
    part = {}
    for s in "BC":
        keys = [(str(c), int(i)) for c, i in zip(z[f"{s}_case"], z[f"{s}_cid"])]
        part[s] = {"keys": keys, **{f: z[f"{s}_{f}"] for f in
                                    ("Po", "Po_cal", "Pn_cal", "ind", "solo")}}
    b, c = part["B"], part["C"]
    nB, nC = len(sp["B"]), len(sp["C"])
    cb = candidates(b["Po"], b["Po_cal"], b["Pn_cal"], b["solo"], b["ind"], b["keys"],
                    meas, policy)
    cc = candidates(c["Po"], c["Po_cal"], c["Pn_cal"], c["solo"], c["ind"], c["keys"],
                    meas, policy)

    print(f"[{T}] risk-benefit curve: certify on B (n={nB}), measured on C (n={nC})")
    rep["curve"] = []
    for dl in CURVE + ["naive"]:
        cfg = naive(cb) if dl == "naive" else choose(cb, certify(cb, nB, dl, ALPHA))
        x = cc[(cfg[0], cfg[2], cfg[1])] if cfg else {"n": 0, "sum": 0.0, "k": 0, "lost": 0}
        y = cb[(cfg[0], cfg[2], cfg[1])] if cfg else {"n": 0, "sum": 0.0, "k": 0}
        row = {"delta": dl, "config": cfg, "B_n": y["n"], "B_sum": y["sum"], "B_k": y["k"],
               "C_n": x["n"], "C_sum": x["sum"], "C_k": x["k"], "C_lost": x["lost"],
               "C_rate": x["k"] / nC, "C_ub95": ub(x["k"], nC)}
        rep["curve"].append(row)
        print(f"  delta={dl!s:6s} {str(cfg or 'rollback'):34s} B: n={y['n']:3d} "
              f"sum={y['sum']:+6.2f} k={y['k']} | C: n={x['n']:3d} sum={x['sum']:+6.2f} "
              f"k={x['k']} ({x['k'] / nC:.3f}, UB {row['C_ub95']:.3f})")

    print(f"[{T}] validity: {N_SPLIT} random halves of B u C")
    keys = b["keys"] + c["keys"]
    cat = {f: np.concatenate([b[f], c[f]]) for f in ("Po", "Po_cal", "Pn_cal", "ind", "solo")}
    pid = sorted({patient_of(s) for s in sp["B"] + sp["C"]})
    HB = [per_patient((d, t, g), cat["Po"], cat["Po_cal"], cat["Pn_cal"], cat["solo"],
                      cat["ind"], keys, meas, policy, pid)
          for d, g in CHAINS for t in CHAIN_GATES]
    H = np.array([h for h, _ in HB])
    Bn = np.array([v for _, v in HB])
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(len(pid)) for _ in range(N_SPLIT)]
    rep["validity"] = []
    for dl in VALID:
        res = {"ltt": [], "naive": []}
        for perm in splits:
            se, ev = perm[: len(pid) // 2], perm[len(pid) // 2:]
            j = ltt_fast(H[:, se], Bn[:, se], dl, ALPHA)
            jn = int(np.argmax(Bn[:, se].sum(1)))
            res["ltt"].append((H[j, ev].mean(), Bn[j, ev].sum()) if j is not None else (0.0, 0.0))
            res["naive"].append((H[jn, ev].mean(), Bn[jn, ev].sum()))
        row = {"delta": dl, "n_select": len(pid) // 2, "n_eval": len(pid) - len(pid) // 2}
        for name, v in res.items():
            risk = np.array([r for r, _ in v])
            ben = np.array([s for _, s in v])
            row[name] = {"violation_rate": float((risk > dl).mean()),
                         "mean_risk": float(risk.mean()), "mean_benefit": float(ben.mean())}
        rep["validity"].append(row)
        print(f"  delta={dl:.2f}: violation LTT {row['ltt']['violation_rate']:.3f} vs NAIVE "
              f"{row['naive']['violation_rate']:.3f} | mean risk {row['ltt']['mean_risk']:.4f}"
              f" vs {row['naive']['mean_risk']:.4f} | benefit {row['ltt']['mean_benefit']:+.2f}"
              f" vs {row['naive']['mean_benefit']:+.2f}")
    (R / f"phase19_eval_{T}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
