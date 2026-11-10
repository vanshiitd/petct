#!/usr/bin/env python3
"""Phase XIX: certified self-improvement on PSMA -- fit on A, certify on B.

See `reports/phase19_psma_selfimprove_plan.md`. For one segmenter (`--tag`):

  A  adapt:   Po_cal (frozen hand-built detector, calibrated on A), the adapted
              new-feature detector (FDG out-of-fold + A, calibrator on 5-fold
              cross-fitted A predictions), the fence (Q3 + 3 IQR on A)
  B  certify: the unchanged Phase XVII family, LTT at delta 0.05, alpha 0.05;
              also the NAIVE and UNCAL selections on the same B rows
  C  tables:  one --pfail-table per arm, restricted to C. No C outcome is read:
              labels and measured outcomes are loaded for A and B cases only.

Writes, all new files, for tag T:
  phase19_{T}_{hb,p19,naive,pseudo,uncal}_C.npz   (pseudo/naive/uncal: primary only)
  phase19_{T}_probs.npz        B and C probabilities, for the post-lock analyses
  phase19_{T}.json             fence, candidates, p-values, chosen configurations
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase11_experience_gate import R, features, fit_stage1, predict_stage1, write_table
from phase4_diag_model import PRED_CLASSES
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import DETS, GUARDS, cal_fit, cal_apply
from phase15_ood import fence
from phase16_combined import union
from phase17_ltt import CHAIN_GATES, candidates, certify, choose

DELTA, ALPHA = 0.05, 0.05
SEED, NFOLD = 2027, 5
SPLIT = Path(__file__).resolve().parent.parent / "reports" / "phase19_split.json"


def rows_of(path: Path, kind: str, cases: set) -> list[dict]:
    return [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["kind"] == kind and r["case"] in cases]


def ab_labels(path: Path, keys, allowed: set) -> np.ndarray:
    """Failure-class labels as `labels_from_diag`, for allowed cases only."""
    lab = {}
    for r in pickle.load(open(path, "rb")):
        if r["case"] not in allowed:
            continue
        cc = {c["component_id"]: {k: 0 for k in PRED_CLASSES} for c in r["components"]}
        for c in r["components"]:
            if c["spurious_class"]:
                cc[c["component_id"]][c["spurious_class"]] = 1
            if c["merging"]:
                cc[c["component_id"]]["merging"] = 1
        for l in r["lesions"]:
            for k in ("under_seg", "over_seg", "boundary", "fragmentation"):
                if l[k]:
                    for cid in l["comp_ids"]:
                        if cid in cc:
                            cc[cid][k] = 1
        for cid, d in cc.items():
            lab[(r["case"], cid)] = d
    assert all(k[0] in allowed for k in keys)
    return np.array([[lab.get(k, {}).get(c, 0) for c in PRED_CLASSES] for k in keys], float)


def ab_measured(path: Path, allowed: set) -> dict:
    """`phase11_experience_gate.measured` for allowed cases only."""
    meas = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["strategy"] == "recover" or r["case"] not in allowed:
            continue
        try:
            cid = int(r["component_id"])
        except ValueError:
            continue
        meas[(r["case"], abs(cid))][r["strategy"]] = r
    return meas


def naive(cands) -> tuple | None:
    """Largest summed measured benefit on B, adopted if that sum is > 0."""
    best = max(cands.items(), key=lambda kv: (kv[1]["sum"], kv[0][2]))
    (d, g, t), c = best
    return (d, t, g) if c["sum"] > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="psma | swin_psma | nnuf0_psma")
    ap.add_argument("--arms", nargs="+", default=["hb", "p19", "naive", "pseudo", "uncal"])
    ap.add_argument("--b-tables", action="store_true",
                    help="Phase XX: instead of the C tables, write the hb/p19 tables for "
                         "B as phase20_{T}_{arm}_B.npz, and nothing else, so the locked "
                         "Phase XIX outputs are never rewritten")
    args = ap.parse_args()
    T = args.tag
    sp = json.loads(SPLIT.read_text())
    A, B, C = set(sp["A"]), set(sp["B"]), set(sp["C"])
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    old, new = features(), features() + NEW

    # FDG out-of-fold rows: the training data every earlier phase used
    frows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    _, Yf = labels(frows, R / "diag_oof.pkl")
    X0f, med0, _ = matrix(frows, old, None)
    X1f, med1, _ = matrix(frows, new, None)

    D = pickle.load(open(R / "phase4_diag_model.pkl", "rb"))
    Dp, Dr = D["pred"], D["recover"]

    def part(cases):
        rs = rows_of(R / f"cand_{T}_ctx.csv", "pred", cases)
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        nc = collections.Counter(r["case"] for r in rs)
        nr = np.array([nc[r["case"]] for r in rs])
        Xd = matrix(rs, Dp["feats"], Dp["median"])[0]
        Po = np.column_stack([Dp["models"][c].predict_proba(Xd)[:, 1] for c in PRED_CLASSES])
        return {"rows": rs, "keys": ks, "nr": nr, "Po": Po,
                "X1": matrix(rs, new, med1)[0]}

    pa, pb, pc = part(A), part(B), part(C)
    Ya = ab_labels(R / f"diag_{T}.pkl", pa["keys"], A)
    Yb = ab_labels(R / f"diag_{T}.pkl", pb["keys"], B)
    meas = ab_measured(R / f"strategy_table_{T}.csv", A | B)

    # ---- adapt on A ------------------------------------------------------
    fz = fence(list(collections.Counter(r["case"] for r in pa["rows"]).values()))
    indA = pa["nr"] <= fz
    Co = cal_fit(pa["Po"][indA], Ya[indA])

    acases = sorted(A)
    rng = np.random.default_rng(SEED)
    fold_of = {c: i % NFOLD for i, c in enumerate(rng.permutation(acases))}
    fa = np.array([fold_of[k[0]] for k in pa["keys"]])
    PnA = np.full_like(Ya, np.nan)
    for j in range(NFOLD):
        tr, te = fa != j, fa == j
        M = fit_stage1(np.vstack([X1f, pa["X1"][tr]]), np.vstack([Yf, Ya[tr]]),
                       np.arange(len(X1f) + int(tr.sum())))
        PnA[te] = predict_stage1(M, pa["X1"][te])
    Cn = cal_fit(PnA[indA], Ya[indA])
    Mn = fit_stage1(np.vstack([X1f, pa["X1"]]), np.vstack([Yf, Ya]),
                    np.arange(len(X1f) + len(Ya)))

    # pseudo-label self-training: no PSMA ground truth
    Ypl = (pa["Po"] >= 0.5).astype(float)
    Mpl = fit_stage1(np.vstack([X1f, pa["X1"]]), np.vstack([Yf, Ypl]),
                     np.arange(len(X1f) + len(Ypl)))

    for p in (pb, pc):
        p["ind"], p["solo"] = p["nr"] <= fz, p["nr"] > 1
        p["Po_cal"] = cal_apply(Co, p["Po"])
        p["Pn_raw"] = predict_stage1(Mn, p["X1"])
        p["Pn_cal"] = cal_apply(Cn, p["Pn_raw"])
        p["Ppl"] = predict_stage1(Mpl, p["X1"])

    # ---- certify / select on B ------------------------------------------
    nB = len(B)
    cands = candidates(pb["Po"], pb["Po_cal"], pb["Pn_cal"], pb["solo"], pb["ind"],
                       pb["keys"], meas, policy)
    cert = certify(cands, nB, DELTA, ALPHA)
    cfg = choose(cands, cert)
    ucands = candidates(pb["Po"], pb["Po"], pb["Pn_raw"], pb["solo"], pb["ind"],
                        pb["keys"], meas, policy)
    ucert = certify(ucands, nB, DELTA, ALPHA)
    ucfg = choose(ucands, ucert)
    ncfg = naive(cands)
    print(f"[{T}] fence {fz:.1f} | rows A {len(pa['keys'])} B {len(pb['keys'])} "
          f"C {len(pc['keys'])} | B patients {nB}")
    for d in DETS:
        for g in GUARDS:
            line = " ".join(f"{t:.2f}:k{cands[(d, g, t)]['k']}"
                            f"{'*' if (d, g, t) in cert else ''}" for t in CHAIN_GATES)
            print(f"  {d:7s} {g:10s} {line}")
    for name, cf, cs in (("P19", cfg, cands), ("UNCAL", ucfg, ucands), ("NAIVE", ncfg, cands)):
        c = cs[(cf[0], cf[2], cf[1])] if cf else None
        print(f"  {name:6s} -> {cf or 'rollback'}"
              + (f"  B: n={c['n']} sum={c['sum']:+.2f} lcb={c['lcb']:+.4f} k={c['k']} "
                 f"lost={c['lost']:.0f}" if c else ""))

    if args.b_tables:
        rr = rows_of(R / f"cand_{T}_pos.csv", "recover", B)
        rk = [(r["case"], int(r["component_id"])) for r in rr]
        Xr = matrix(rr, Dr["feats"], Dr["median"])[0]
        prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)
        for a, cf in (("hb", None), ("p19", cfg)):
            P, act, added = union(cf, pb["Po"], pb["Po_cal"], pb["Pn_cal"], pb["solo"], pb["ind"])
            write_table(R / f"phase20_{T}_{a}_B.npz", pb["keys"], P, act, rk, prs)
            print(f"  B {a}: acted {int(act.sum())} added {int(added.sum())}")
        return

    # ---- C tables (actions only; no C outcome is read) -------------------
    rr = rows_of(R / f"cand_{T}_pos.csv", "recover", C)
    rk = [(r["case"], int(r["component_id"])) for r in rr]
    Xr = matrix(rr, Dr["feats"], Dr["median"])[0]
    prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)
    allC = np.ones(len(pc["keys"]), bool)
    arms = {
        "hb": union(None, pc["Po"], pc["Po_cal"], pc["Pn_cal"], pc["solo"], pc["ind"]),
        "p19": union(cfg, pc["Po"], pc["Po_cal"], pc["Pn_cal"], pc["solo"], pc["ind"]),
        "naive": union(ncfg, pc["Po"], pc["Po_cal"], pc["Pn_cal"], pc["solo"], pc["ind"]),
        "uncal": union(ucfg, pc["Po"], pc["Po"], pc["Pn_raw"], pc["solo"], pc["ind"]),
        "pseudo": union(("new_cal", 0.90, "none"), pc["Po"], pc["Po"], pc["Ppl"],
                        pc["solo"], allC),
    }
    counts = {}
    for a in args.arms:
        P, act, added = arms[a]
        write_table(R / f"phase19_{T}_{a}_C.npz", pc["keys"], P, act, rk, prs)
        counts[a] = {"acted": int(act.sum()), "added": int(added.sum())}
    print(f"  C counts: {counts}")

    np.savez_compressed(
        R / f"phase19_{T}_probs.npz",
        **{f"{n}_{f}": np.asarray(p[f]) for n, p in (("B", pb), ("C", pc))
           for f in ("Po", "Po_cal", "Pn_raw", "Pn_cal", "Ppl", "ind", "solo")},
        B_case=np.array([k[0] for k in pb["keys"]]), B_cid=np.array([k[1] for k in pb["keys"]]),
        C_case=np.array([k[0] for k in pc["keys"]]), C_cid=np.array([k[1] for k in pc["keys"]]))
    rep = {"plan": "reports/phase19_psma_selfimprove_plan.md", "tag": T, "delta": DELTA,
           "alpha": ALPHA, "n_B": nB, "fence": fz,
           "rows": {"A": len(pa["keys"]), "B": len(pb["keys"]), "C": len(pc["keys"])},
           "p19": cfg, "uncal": ucfg, "naive": ncfg, "pseudo": ("new_cal", 0.90, "none"),
           "n_certified": len(cert), "n_certified_uncal": len(ucert),
           "candidates": [{"det": d, "guard": g, "t": t, **c,
                           "p": float(binom.cdf(c["k"], nB, DELTA)),
                           "certified": (d, g, t) in cert}
                          for (d, g, t), c in cands.items()],
           "candidates_uncal": [{"det": d, "guard": g, "t": t, **c,
                                 "certified": (d, g, t) in ucert}
                                for (d, g, t), c in ucands.items()],
           "C_counts": counts}
    (R / f"phase19_{T}.json").write_text(json.dumps(rep, indent=1, default=float),
                                         encoding="utf-8")
    print(f"written {R / f'phase19_{T}.json'}")


if __name__ == "__main__":
    main()
