#!/usr/bin/env python3
"""Phase XII: KL-anchored policy improvement over the hand-built policy.

See `reports/phase12_anchored_policy_plan.md`. The hand-built policy is the
reference policy pi_ref and every one of its actions is kept. The learned
policy may only *add* actions below the 0.90 gate, where the Phase XI harm
model (the verifier) is confident: act iff pmax >= 0.90, or
g <= pmax < 0.90 and harm <= h. (g, h) is promoted on training folds only
under CAVE-PET causal-utility criteria, else the policy rolls back to pi_ref.

Reuses `phase11_experience_gate` for stage 1, context, the harm model and
the table writer, so the nesting is the one verified there (stage 1
reproduces audit_crossfit.npz exactly).

Writes, all new files:
  phase12_anchored_oof.npz, phase12_anchored_{val,test}.npz
  phase12_anchored.json   selection, held-out additions, faithfulness,
                          experience curve, calibration
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase4_diag_model import PRED_CLASSES, labels_from_diag
from phase11_experience_gate import (R, SPLITS, FOLDS, BASE_GATE, features, measured,
                                     labels_for, fit_stage1, predict_stage1,
                                     stage1_crossfit, context, harm_design, fit_harm,
                                     harm_crossfit, write_table, calibration, cp_ub)
from q_model import patient_of

G_GRID = [0.80, 0.70, 0.60, 0.50]
H_GRID = [0.01, 0.02, 0.05, 0.10]
MIN_N = 20
MAX_HARM_UB = 0.05
MAX_LOST = 2
MIN_POS_SUM = -0.5
NBOOT = 2000
SEED = 2026
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
N_SHUFFLE = 20


def added_idx(idx, outs, pmax, ph, g, h):
    return [i for i in idx if outs[i] is not None and g <= pmax[i] < BASE_GATE
            and ph[i] <= h]


def stats(idx, outs, keys):
    """Summed and mean outcome, patient-clustered LCB, harm UB, lesions lost."""
    n = len(idx)
    if n == 0:
        return {"n": 0, "sum": 0.0, "lost": 0.0, "harm": 0, "harm_ub": 1.0,
                "lcb": float("nan"), "mean": float("nan"), "pos_sum": 0.0, "neg_sum": 0.0}
    by_p = collections.defaultdict(lambda: [0.0, 0])
    for i in idx:
        p = patient_of(keys[i][0])
        by_p[p][0] += outs[i]["y"]
        by_p[p][1] += 1
    s = np.array([v[0] for v in by_p.values()])
    c = np.array([v[1] for v in by_p.values()])
    rng = np.random.default_rng(SEED)
    m = len(s)
    boot = np.array([s[ix].sum() / c[ix].sum()
                     for ix in (rng.integers(0, m, m) for _ in range(NBOOT))])
    k = sum(outs[i]["harm"] for i in idx)
    return {"n": n, "sum": float(s.sum()), "mean": float(s.sum() / c.sum()),
            "lcb": float(np.percentile(boot, 2.5)),
            "harm": int(k), "harm_ub": cp_ub(k, n),
            "lost": float(sum(outs[i]["lost"] for i in idx)),
            "pos_sum": float(sum(outs[i]["y"] for i in idx if outs[i]["pos"])),
            "neg_sum": float(sum(outs[i]["y"] for i in idx if not outs[i]["pos"]))}


def promotable(st):
    return (st["n"] >= MIN_N and st["lcb"] > 0 and st["harm_ub"] <= MAX_HARM_UB
            and st["lost"] <= MAX_LOST and st["pos_sum"] >= MIN_POS_SUM)


def select(idx, outs, pmax, ph, keys):
    best, table = None, []
    for g in G_GRID:
        for h in H_GRID:
            st = stats(added_idx(idx, outs, pmax, ph, g, h), outs, keys)
            ok = promotable(st)
            table.append({"g": g, "h": h, **st, "promotable": ok})
            if ok:
                key = (st["sum"], g, -h)
                if best is None or key > best[0]:
                    best = (key, g, h)
    return (best[1], best[2]) if best else (None, None), table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, default=R / "cand_cv_oof_pos.csv")
    ap.add_argument("--diag", type=Path, default=R / "diag_oof.pkl")
    ap.add_argument("--crossfit", type=Path, default=R / "audit_crossfit.npz")
    ap.add_argument("--diagpkl", type=Path, default=R / "phase4_diag_model.pkl")
    ap.add_argument("--table", type=Path, nargs="+",
                    default=[R / "strategy_table_oof.csv", R / "strategy6_oof.csv"])
    args = ap.parse_args()

    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    feats = features()
    rows, _, _, _ = load(args.cand, "pred")
    X, _, _ = matrix(rows, feats, None)
    keys = [(r["case"], int(r["component_id"])) for r in rows]
    fold = np.array([owner.get(r["case"], -1) for r in rows])
    lab, _ = labels_from_diag(args.diag)
    Y = np.array([[lab.get(k, {}).get(c, 0) for c in PRED_CLASSES] for k in keys], float)
    meas = measured(args.table)
    z = np.load(args.crossfit, allow_pickle=True)
    Pcf_map = {(str(c), int(i)): p for c, i, p in
               zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
    P_cf = np.array([Pcf_map[k] for k in keys])
    rec_keys = [(str(c), int(i)) for c, i in zip(z["rec_keys_case"], z["rec_keys_cid"])]
    rec_cf = np.asarray(z["rec_crossfit"], float)
    pmax_cf = P_cf.max(1)
    outs_cf = labels_for(keys, P_cf, meas, policy)

    report = {"plan": "reports/phase12_anchored_policy_plan.md",
              "grid": {"g": G_GRID, "h": H_GRID}, "outer": []}
    act = pmax_cf >= BASE_GATE                     # pi_ref, always kept
    ph_held = np.full(len(rows), np.nan)
    held_added = []
    rng = np.random.default_rng(SEED)
    for k in FOLDS:
        T = [f for f in FOLDS if f != k]
        tr_idx = np.where(np.isin(fold, T))[0]
        te = fold == k
        te_idx = list(np.where(te)[0])
        P_in = stage1_crossfit(X, Y, fold, T)
        P_in[te] = predict_stage1(fit_stage1(X, Y, tr_idx), X[te])
        assert np.allclose(P_in[te], P_cf[te]), "stage 1 drifted from audit_crossfit"
        Z = harm_design(X, P_in, context(rows, P_in))
        pmax = P_in.max(1)
        outs = labels_for(keys, P_in, meas, policy)
        ph = harm_crossfit(Z, outs, pmax, fold, T)
        ph[te] = fit_harm(Z, outs, pmax, tr_idx).predict_proba(Z[te])[:, 1]
        ph_held[te] = ph[te]

        (g, h), table = select(list(tr_idx), outs, pmax, ph, keys)
        entry = {"fold": k, "g": g, "h": h, "train_table": table}
        if g is None:
            print(f"outer fold {k}: nothing promotable -> roll back to pi_ref")
        else:
            add = added_idx(te_idx, outs, pmax, ph, g, h)
            for i in add:
                act[i] = True
            held_added += add
            st = stats(add, outs, keys)
            tr_st = next(t for t in table if t["g"] == g and t["h"] == h)
            entry["heldout_added"] = st
            print(f"outer fold {k}: promoted g={g}, h={h}  (train: n={tr_st['n']} "
                  f"sum={tr_st['sum']:+.2f} LCB={tr_st['lcb']:+.4f} "
                  f"harm_ub={tr_st['harm_ub']:.3f} lost={tr_st['lost']:.0f})")
            print(f"   held-out additions: n={st['n']} sum={st['sum']:+.2f} "
                  f"(tumour-free {st['neg_sum']:+.2f}, positive {st['pos_sum']:+.2f}) "
                  f"lost={st['lost']:.0f} harm={st['harm']}")

            # experience faithfulness: shuffle the verifier among eligible rows
            elig = [i for i in te_idx if outs[i] is not None and g <= pmax[i] < BASE_GATE]
            sh = []
            for _ in range(N_SHUFFLE):
                phs = ph.copy()
                phs[elig] = rng.permutation(ph[elig])
                sh.append(stats(added_idx(elig, outs, pmax, phs, g, h), outs, keys))
            entry["faithfulness_shuffled"] = {
                "mean_sum": float(np.mean([s["sum"] for s in sh])),
                "mean_lost": float(np.mean([s["lost"] for s in sh])),
                "mean_n": float(np.mean([s["n"] for s in sh]))}
            print(f"   shuffled verifier (x{N_SHUFFLE}): sum={entry['faithfulness_shuffled']['mean_sum']:+.2f} "
                  f"lost={entry['faithfulness_shuffled']['mean_lost']:.1f}")

            # self-improvement curve: harm model on a fraction of experience
            pats = sorted({patient_of(keys[i][0]) for i in tr_idx})
            curve = []
            for frac in FRACTIONS:
                sub_p = set(np.random.default_rng(SEED).choice(
                    pats, max(1, int(round(frac * len(pats)))), replace=False))
                sub = [i for i in tr_idx if patient_of(keys[i][0]) in sub_p]
                phf = ph.copy()
                phf[te] = fit_harm(Z, outs, pmax, sub).predict_proba(Z[te])[:, 1]
                s = stats(added_idx(te_idx, outs, pmax, phf, g, h), outs, keys)
                curve.append({"fraction": frac, "sum": s["sum"], "lost": s["lost"], "n": s["n"]})
            entry["experience_curve"] = curve
            print("   experience curve: " + "  ".join(
                f"{c['fraction']:.0%}: {c['sum']:+.2f}/{c['lost']:.0f}" for c in curve))
        report["outer"].append(entry)

    tot = stats(held_added, outs_cf, keys)
    report["heldout_added_total"] = tot
    print(f"\nheld-out additions, all folds: {tot}")
    sh_tot = {m: sum(e.get("faithfulness_shuffled", {}).get(m, 0) for e in report["outer"])
              for m in ("mean_sum", "mean_lost", "mean_n")}
    report["faithfulness_total"] = {"learned": {"sum": tot["sum"], "lost": tot["lost"]},
                                    "shuffled": sh_tot}
    print(f"faithfulness: learned verifier sum={tot['sum']:+.2f} lost={tot['lost']:.0f}  |  "
          f"shuffled sum={sh_tot['mean_sum']:+.2f} lost={sh_tot['mean_lost']:.1f}")
    curve_tot = {f: {m: sum(c[m] for e in report["outer"] for c in e.get("experience_curve", [])
                            if c["fraction"] == f) for m in ("sum", "lost")}
                 for f in FRACTIONS}
    report["experience_curve_total"] = curve_tot
    print("experience curve, all folds: " + "  ".join(
        f"{f:.0%}: {v['sum']:+.2f}/{v['lost']:.0f}" for f, v in curve_tot.items()))

    elig_all = np.array([o is not None and 0.5 <= pm < BASE_GATE
                         for o, pm in zip(outs_cf, pmax_cf)])
    yh = np.array([o["harm"] if o is not None else 0 for o in outs_cf], float)
    report["calibration_below_gate"] = calibration(ph_held[elig_all], yh[elig_all])
    c = report["calibration_below_gate"]
    print(f"harm-model calibration below the gate: ECE {c['ece']:.4f} Brier {c['brier']:.4f} "
          f"prev {c['prevalence']:.4f}")
    print(f"acted components out-of-fold: {int(act.sum())} (pi_ref {int((pmax_cf >= BASE_GATE).sum())})")
    write_table(R / "phase12_anchored_oof.npz", keys, P_cf, act, rec_keys, rec_cf)

    # ---- production -----------------------------------------------------
    Z_cf = harm_design(X, P_cf, context(rows, P_cf))
    ph_cv = harm_crossfit(Z_cf, outs_cf, pmax_cf, fold, list(FOLDS))
    all_idx = list(np.where(fold >= 0)[0])
    (g, h), table = select(all_idx, outs_cf, pmax_cf, ph_cv, keys)
    report["production"] = {"g": g, "h": h, "table": table}
    print(f"\nproduction: g={g}, h={h}" + ("" if g is not None else " (roll back to pi_ref)"))
    harm_model = fit_harm(Z_cf, outs_cf, pmax_cf, all_idx)
    D = pickle.load(open(args.diagpkl, "rb"))
    Dp, Dr = D["pred"], D["recover"]
    assert Dp["feats"] == feats
    for split in ("val", "test"):
        cand = R / f"cand_cv_{split}_pos.csv"
        rs, _, _, _ = load(cand, "pred")
        Xs, _, _ = matrix(rs, Dp["feats"], Dp["median"])
        Ps = np.column_stack([Dp["models"][c].predict_proba(Xs)[:, 1] for c in PRED_CLASSES])
        pm = Ps.max(1)
        acts = pm >= BASE_GATE
        if g is not None:
            phs = harm_model.predict_proba(harm_design(Xs, Ps, context(rs, Ps)))[:, 1]
            acts |= (pm >= g) & (pm < BASE_GATE) & (phs <= h)
        rr, _, _, _ = load(cand, "recover")
        Xr, _, _ = matrix(rr, Dr["feats"], Dr["median"])
        prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)
        write_table(R / f"phase12_anchored_{split}.npz",
                    [(r["case"], int(r["component_id"])) for r in rs], Ps, acts,
                    [(r["case"], int(r["component_id"])) for r in rr], prs)
        report[f"production_{split}"] = {"acted": int(acts.sum()),
                                         "pi_ref": int((pm >= BASE_GATE).sum())}

    (R / "phase12_anchored.json").write_text(json.dumps(report, indent=1, default=float),
                                             encoding="utf-8")
    print(f"written {R / 'phase12_anchored.json'}")


if __name__ == "__main__":
    main()
