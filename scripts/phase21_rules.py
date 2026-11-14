#!/usr/bin/env python3
"""Phase XXI, Parts A and B: selection rules compared in the validity study,
and certification under a known tracer or vendor shift.

See `reports/phase21_baselines_shift_safeguard_plan.md`. Post hoc and
retrospective: it resamples the Phase XIX per-patient harm and benefit tables
of B u C. No model is run and no earlier file is written.

Writes phase21_rules.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import patient_of
from phase11_experience_gate import R, measured
from phase17_ltt import CHAIN_GATES
from phase17b_lifelong import CHAINS, per_patient, ltt_fast
from phase19_selfimprove import ALPHA, SPLIT

T = "psma"
DELTAS = [0.03, 0.05, 0.10, 0.15, 0.20]
N_SPLIT, SEED_A, SEED_B = 1000, 2027, 2028
TSV = Path(r"D:\data\psma_nifti\PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv")
NG = len(CHAIN_GATES)
RULES = ["NAIVE", "PLUGIN", "ZERO", "UNCORR", "CRC", "BONF-88", "CSI"]


def tables():
    """H, Bn (88 candidates x patients of B u C) exactly as phase19_eval.py."""
    sp = json.loads(SPLIT.read_text())
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    z = np.load(R / f"phase19_{T}_probs.npz")
    meas = measured([R / f"strategy_table_{T}.csv"])
    part = {}
    for s in "BC":
        keys = [(str(c), int(i)) for c, i in zip(z[f"{s}_case"], z[f"{s}_cid"])]
        part[s] = {"keys": keys, **{f: z[f"{s}_{f}"] for f in
                                    ("Po", "Po_cal", "Pn_cal", "ind", "solo")}}
    b, c = part["B"], part["C"]
    keys = b["keys"] + c["keys"]
    cat = {f: np.concatenate([b[f], c[f]]) for f in ("Po", "Po_cal", "Pn_cal", "ind", "solo")}
    pid = sorted({patient_of(s) for s in sp["B"] + sp["C"]})
    HB = [per_patient((d, t, g), cat["Po"], cat["Po_cal"], cat["Pn_cal"], cat["solo"],
                      cat["ind"], keys, meas, policy, pid)
          for d, g in CHAINS for t in CHAIN_GATES]
    case_of = {patient_of(s): s for s in sp["B"] + sp["C"]}
    return (np.array([h for h, _ in HB]), np.array([v for _, v in HB]), pid,
            [case_of[p] for p in pid])


def best(ok, ben):
    """Qualifying candidate with the largest selection-half benefit (first on ties)."""
    idx = np.where(ok)[0]
    return int(idx[np.argmax(ben[idx])]) if len(idx) else None


def crc_ok(k, n, delta):
    """Per chain: the longest prefix of gates whose running-max k keeps
    (k + 1) / (n + 1) <= delta."""
    ok = np.zeros(len(k), bool)
    for c in range(len(CHAINS)):
        km = np.maximum.accumulate(k[c * NG:(c + 1) * NG])
        good = (km + 1) / (n + 1) <= delta
        m = NG if good.all() else int(np.argmin(good))
        ok[c * NG:c * NG + m] = True
    return ok


def select(Hs, Bs, delta):
    """Every rule's pick on a selection set: {rule: (index or None, n qualifying)}."""
    n = Hs.shape[1]
    k = Hs.sum(1)
    ben = Bs.sum(1)
    p = binom.cdf(k, n, delta)
    oks = {"NAIVE": np.ones(len(k), bool), "PLUGIN": k / n <= delta, "ZERO": k == 0,
           "UNCORR": p <= ALPHA, "CRC": crc_ok(k, n, delta), "BONF-88": p <= ALPHA / len(k)}
    out = {r: (best(ok, ben), int(ok.sum())) for r, ok in oks.items()}
    # CSI: the unchanged Phase XVII-b routine; its qualifying count is recomputed here
    cert = 0
    for c in range(len(CHAINS)):
        for g in range(NG):
            if p[c * NG + g] > ALPHA / len(CHAINS):
                break
            cert += 1
    out["CSI"] = (ltt_fast(Hs, Bs, delta, ALPHA), cert)
    # CRC inside each single chain, where its own guarantee applies
    out["_crc_chain"] = []
    for c in range(len(CHAINS)):
        ok = np.zeros(len(k), bool)
        ok[c * NG:(c + 1) * NG] = oks["CRC"][c * NG:(c + 1) * NG]
        out["_crc_chain"].append(best(ok, ben))
    return out


def cp(x, n):
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return [lo, hi]


def summary(risk, ben, roll, nq, delta):
    risk, ben = np.asarray(risk), np.asarray(ben)
    v = int((risk > delta).sum())
    return {"violation_rate": v / len(risk), "violation_ci95": cp(v, len(risk)),
            "mean_risk": float(risk.mean()), "p95_risk": float(np.percentile(risk, 95)),
            "mean_benefit": float(ben.mean()), "rollback_freq": float(np.mean(roll)),
            "mean_qualifying": float(np.mean(nq))}


def outcome(j, H, Bn, ev):
    return (0.0, 0.0) if j is None else (float(H[j, ev].mean()), float(Bn[j, ev].sum()))


def part_a(H, Bn):
    n = H.shape[1]
    rng = np.random.default_rng(SEED_A)
    splits = [rng.permutation(n) for _ in range(N_SPLIT)]
    rows = []
    for dl in DELTAS:
        acc = {r: {"risk": [], "ben": [], "roll": [], "nq": []} for r in RULES}
        chain_risk = []
        for perm in splits:
            se, ev = perm[: n // 2], perm[n // 2:]
            sel = select(H[:, se], Bn[:, se], dl)
            for r in RULES:
                j, nq = sel[r]
                risk, ben = outcome(j, H, Bn, ev)
                a = acc[r]
                a["risk"].append(risk); a["ben"].append(ben)
                a["roll"].append(j is None); a["nq"].append(nq)
            chain_risk.append([outcome(j, H, Bn, ev)[0] for j in sel["_crc_chain"]])
        row = {"delta": dl, "n_select": n // 2, "n_eval": n - n // 2,
               "rules": {r: summary(a["risk"], a["ben"], a["roll"], a["nq"], dl)
                         for r, a in acc.items()}}
        cr = np.array(chain_risk)
        row["crc_per_chain"] = {"mean_risk": cr.mean(0).tolist(),
                                "violation_rate": (cr > dl).mean(0).tolist()}
        rows.append(row)
        print(f"delta {dl:.2f}")
        for r in RULES:
            s = row["rules"][r]
            print(f"  {r:8s} viol {s['violation_rate']:.3f} [{s['violation_ci95'][0]:.3f}, "
                  f"{s['violation_ci95'][1]:.3f}] risk {s['mean_risk']:.4f} "
                  f"p95 {s['p95_risk']:.3f} benefit {s['mean_benefit']:+6.2f} "
                  f"rollback {s['rollback_freq']:.2f} qualifying {s['mean_qualifying']:.1f}")
        print(f"  CRC per chain: mean risk {np.round(cr.mean(0), 4).tolist()}")
    return rows


def part_b(H, Bn, cases):
    meta = {}
    for r in csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"):
        meta[(r["case_identifier_number"], r["study_date"].replace("-", ""))] = r

    def info(c):
        pid, date, _ = c.rsplit("_", 2)
        return meta[(pid, date)]

    tracer = np.array([info(c)["pet_radionuclide"] for c in cases])
    man = np.array([info(c)["Manufacturer"] for c in cases])
    names = sorted(set(man), key=lambda v: -(man == v).sum())  # vendor 1 = larger
    vendor = np.array([f"vendor {names.index(m) + 1}" for m in man])
    dirs = [("18F", "68Ga", tracer), ("68Ga", "18F", tracer),
            ("vendor 1", "vendor 2", vendor), ("vendor 2", "vendor 1", vendor)]
    out = []
    for src, tgt, lab in dirs:
        s_idx, t_idx = np.where(lab == src)[0], np.where(lab == tgt)[0]
        rng = np.random.default_rng(SEED_B)
        draws = [rng.permutation(s_idx) for _ in range(N_SPLIT)]
        res = {"source": src, "target": tgt, "n_select": len(s_idx) // 2,
               "n_control": len(s_idx) - len(s_idx) // 2, "n_shifted": len(t_idx), "delta": []}
        print(f"{src} -> {tgt}: select {res['n_select']}, control {res['n_control']}, "
              f"shifted {res['n_shifted']}")
        for dl in DELTAS:
            acc = {r: {"c": [], "s": [], "roll": []} for r in RULES}
            for perm in draws:
                se, co = perm[: len(s_idx) // 2], perm[len(s_idx) // 2:]
                sel = select(H[:, se], Bn[:, se], dl)
                for r in RULES:
                    j = sel[r][0]
                    acc[r]["c"].append(outcome(j, H, Bn, co)[0])
                    acc[r]["s"].append(outcome(j, H, Bn, t_idx)[0])
                    acc[r]["roll"].append(j is None)
            row = {"delta": dl, "rules": {}}
            for r, a in acc.items():
                c, s = np.array(a["c"]), np.array(a["s"])
                vc, vs = int((c > dl).sum()), int((s > dl).sum())
                row["rules"][r] = {
                    "control": {"violation_rate": vc / N_SPLIT, "violation_ci95": cp(vc, N_SPLIT),
                                "mean_risk": float(c.mean())},
                    "shifted": {"violation_rate": vs / N_SPLIT, "violation_ci95": cp(vs, N_SPLIT),
                                "mean_risk": float(s.mean()), "max_risk": float(s.max())},
                    "rollback_freq": float(np.mean(a["roll"]))}
            res["delta"].append(row)
            for r in ("NAIVE", "CRC", "CSI"):
                x = row["rules"][r]
                print(f"  delta {dl:.2f} {r:6s} control viol {x['control']['violation_rate']:.3f} "
                      f"risk {x['control']['mean_risk']:.4f} | shifted viol "
                      f"{x['shifted']['violation_rate']:.3f} risk {x['shifted']['mean_risk']:.4f}"
                      f" | rollback {x['rollback_freq']:.2f}")
        out.append(res)
    return out


def main() -> None:
    H, Bn, pid, cases = tables()
    print(f"H {H.shape}, harmed patients per candidate: min {H.sum(1).min()} "
          f"max {H.sum(1).max()}")
    # how often a patient's harm indicator falls as the gate is lowered
    Hc = H.reshape(len(CHAINS), NG, -1)
    nonmono = int((np.diff(Hc, axis=1) < 0).sum())
    print(f"non-monotone (harm 1 -> 0 along a chain): {nonmono} of "
          f"{Hc[:, 1:].size} gate steps x patients")

    a = part_a(H, Bn)
    ref = {r["delta"]: r for r in
           json.loads((R / f"phase19_eval_{T}.json").read_text())["validity"]}
    for row in a:
        if row["delta"] in ref:
            for mine, theirs in (("CSI", "ltt"), ("NAIVE", "naive")):
                for k in ("violation_rate", "mean_risk", "mean_benefit"):
                    d = abs(row["rules"][mine][k] - ref[row["delta"]][theirs][k])
                    assert d < 1e-12, (row["delta"], mine, k, d)
    print("reproduction check passed: CSI and NAIVE equal phase19_eval_psma.json")

    b = part_b(H, Bn, cases)
    rep = {"plan": "reports/phase21_baselines_shift_safeguard_plan.md", "alpha": ALPHA,
           "n_candidates": int(H.shape[0]), "n_patients": int(H.shape[1]),
           "nonmonotone_steps": nonmono, "part_a": a, "part_b": b}
    (R / "phase21_rules.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"written {R / 'phase21_rules.json'}")


if __name__ == "__main__":
    main()
