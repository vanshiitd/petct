#!/usr/bin/env python3
"""Phase XX: certify the segmenter switch on B, then evaluate every arm on C.

See `reports/phase20_segmenter_selfimprove_plan.md`.

  --stage certify  B only. Candidates FT (raw fine-tuned ensemble, the base of
                   the FT+HB run) and FT+HB, each against the deployed P19, with
                   the Phase XIX harm definition. Exact binomial at alpha/2 per
                   candidate, at the primary delta and the dial. Writes
                   phase20_certify.json. No C file is opened.
  --stage eval     C, after the lock. E1 (DEP - M0), E2 (DEP - P19), E3 (harm of
                   DEP vs P19), the dial on C, and the descriptive table of all
                   arms. Writes phase20_eval.json.

Runs (phase5_agent.py out-prefixes, all in D:\\data\\reports):
  B: p20B_psma_p19 (base M0, final P19), p20B_ft_hb (base FT, final FT+HB)
  C: p19_psma_p19 (base M0, final P19), p20_ft_hb (base FT, final FT+HB),
     p20_ft_p20 (final FT+P20)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice

R = Path(r"D:\data\reports")
DELTA, ALPHA = 0.05, 0.05
DIAL = [0.05, 0.10, 0.15, 0.20]
CANDS = ("FT", "FT+HB")


def ub(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))


def harmed(x: dict, ref: dict, gt: dict) -> list[str]:
    """Phase XIX harm: a lesion hit lost, or tumour-positive Dice down > 0.01."""
    assert set(x) == set(ref)
    return [c for c in sorted(x) if x[c]["n_hit"] < ref[c]["n_hit"]
            or (gt[c]["n_gt"] > 0 and x[c]["dice"] - ref[c]["dice"] < -0.01)]


def pos_dice(x: dict) -> float:
    return float(np.nanmean([v["dice"] for v in x.values() if v["n_gt"] > 0]))


def runs(split: str) -> dict:
    p = "p20B" if split == "B" else "p20"
    p19 = "p20B_psma_p19" if split == "B" else "p19_psma_p19"
    out = {"M0": read(R / p19, "base"), "P19": read(R / p19, "final"),
           "FT": read(R / f"{p}_ft_hb", "base"), "FT+HB": read(R / f"{p}_ft_hb", "final")}
    if split == "C":
        out["FT+P20"] = read(R / "p20_ft_p20", "final")
    for v in out.values():
        assert set(v) == set(out["M0"])
    return out


def certify() -> None:
    a = runs("B")
    n = len(a["P19"])
    rep = {"n_B": n, "alpha_each": ALPHA / len(CANDS), "dial": []}
    print(f"[B] n = {n} | mean Dice on tumour-positive: "
          + " ".join(f"{k} {pos_dice(v):.4f}" for k, v in a.items()))
    ks = {c: len(harmed(a[c], a["P19"], a["M0"])) for c in CANDS}
    for dl in DIAL:
        row = {"delta": dl, "cands": {}}
        for c in CANDS:
            p = float(binom.cdf(ks[c], n, dl))
            row["cands"][c] = {"k": ks[c], "p": p, "certified": p < ALPHA / len(CANDS),
                               "dice_B": pos_dice(a[c])}
        ok = [c for c in CANDS if row["cands"][c]["certified"]]
        row["DEP"] = max(ok, key=lambda c: row["cands"][c]["dice_B"]) if ok else "P19"
        rep["dial"].append(row)
        print(f"  delta {dl:.2f}: " + " | ".join(
            f"{c} k={ks[c]} p={row['cands'][c]['p']:.4f}{'*' if row['cands'][c]['certified'] else ''}"
            for c in CANDS) + f"  -> DEP {row['DEP']}")
    rep["DEP"] = rep["dial"][0]["DEP"]
    (R / "phase20_certify.json").write_text(json.dumps(rep, indent=1))
    print(f"primary (delta {DELTA}) DEP = {rep['DEP']}")


def evaluate() -> None:
    cert = json.loads((R / "phase20_certify.json").read_text())
    a = runs("C")
    n = len(a["P19"])
    gt = a["M0"]
    rep = {"n_C": n, "DEP": cert["DEP"], "arms": {}}
    print(f"[C] n = {n}")
    print(f"  {'arm':7s} {'Dice':>7s} {'FP mL':>8s} {'FN mL':>7s} {'hits':>5s} "
          f"{'dDice vs M0 [95%]':>28s} {'harm vs P19':>12s}")
    for k, v in a.items():
        cs = sorted(v)
        d0 = paired_dice(v, gt)
        h = harmed(v, a["P19"], gt)
        rep["arms"][k] = {
            "dice_pos": pos_dice(v),
            "fp_ml": float(np.mean([v[c]["fp_ml"] for c in cs])),
            "fn_ml": float(np.mean([v[c]["fn_ml"] for c in cs])),
            "hits": int(sum(v[c]["n_hit"] for c in cs)),
            "neg_cleared": int(sum(v[c]["n_fp_comp"] == 0 for c in cs if gt[c]["n_gt"] == 0)),
            "vs_M0": d0, "vs_P19": paired_dice(v, a["P19"]),
            "harmed_vs_P19": len(h), "harmed_cases": h}
        print(f"  {k:7s} {rep['arms'][k]['dice_pos']:7.4f} {rep['arms'][k]['fp_ml']:8.1f} "
              f"{rep['arms'][k]['fn_ml']:7.2f} {rep['arms'][k]['hits']:5d} "
              f"{d0['delta']:+.4f} [{d0['lo']:+.4f}, {d0['hi']:+.4f}] p={d0['p']:.4f} "
              f"{len(h):5d}/{n}")

    dep = cert["DEP"]
    e1, e2 = rep["arms"][dep]["vs_M0"], rep["arms"][dep]["vs_P19"]
    k = rep["arms"][dep]["harmed_vs_P19"]
    # Holm over E1 and E2; paired_dice p is two-sided, which is conservative here
    ps = sorted([("E1", e1["p"]), ("E2", e2["p"])], key=lambda t: t[1])
    met, stop = {}, False
    for i, (name, p) in enumerate(ps):
        d = (e1 if name == "E1" else e2)["delta"]
        met[name] = (not stop) and p < 0.025 / (2 - i) and d > 0
        stop = stop or not met[name]
    pe3 = float(binom.cdf(k, n, DELTA))
    rep["E1"] = {**e1, "met": met["E1"]}
    rep["E2"] = {**e2, "met": met["E2"]}
    rep["E3"] = {"k": k, "n": n, "p": pe3, "ub95": ub(k, n), "met": pe3 < ALPHA}
    print(f"  DEP = {dep}")
    print(f"  E1 DEP-M0  {e1['delta']:+.4f} [{e1['lo']:+.4f}, {e1['hi']:+.4f}] p={e1['p']:.4f} "
          f"met={met['E1']}")
    print(f"  E2 DEP-P19 {e2['delta']:+.4f} [{e2['lo']:+.4f}, {e2['hi']:+.4f}] p={e2['p']:.4f} "
          f"met={met['E2']}")
    print(f"  E3 harmed {k}/{n} p={pe3:.4f} UB {ub(k, n):.3f} met={pe3 < ALPHA}")

    rep["dial_C"] = []
    for row in cert["dial"]:
        d = row["DEP"]
        kk = rep["arms"][d]["harmed_vs_P19"]
        rep["dial_C"].append({"delta": row["delta"], "DEP": d, "k": kk, "rate": kk / n,
                              "ub95": ub(kk, n), "dDice_vs_M0": rep["arms"][d]["vs_M0"]["delta"]})
        print(f"  dial delta {row['delta']:.2f}: {d:6s} harmed {kk}/{n} ({kk / n:.3f}, "
              f"UB {ub(kk, n):.3f}) dDice vs M0 {rep['arms'][d]['vs_M0']['delta']:+.4f}")
    (R / "phase20_eval.json").write_text(json.dumps(rep, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("certify", "eval"), required=True)
    certify() if ap.parse_args().stage == "certify" else evaluate()


if __name__ == "__main__":
    main()
