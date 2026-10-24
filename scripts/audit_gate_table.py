#!/usr/bin/env python3
"""Audit R1(a) and R1(e): the p_failure gate, in-sample against cross-fitted.

Both tables are recomputed from the measured strategy table rather than from an
agent run, which is exactly equivalent for a fixed strategy and needs no GPU or
re-simulation: every row of `strategy_table_oof.csv` is the measured effect of
applying one strategy to one component, and the gate only decides which
components are eligible.

  R1(a)  topology_remove by p_failure bin: mean d_dice, lesions lost
  R1(e)  the gated population -- precision and mean effect of removal, the
         "98%" figure in final_eval.md section 4

Writes `audit_gate_table.json`. Nothing existing is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BINS = [(0.0, 0.60), (0.60, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
BIN_LAB = ["<0.60", "0.60-0.80", "0.80-0.90", "0.90-0.95", ">=0.95"]
THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossfit", type=Path,
                    default=Path(r"D:\data\reports\audit_crossfit.npz"))
    ap.add_argument("--table", type=Path,
                    default=Path(r"D:\data\reports\strategy_table_oof.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_gate_table.json"))
    args = ap.parse_args()

    z = np.load(args.crossfit, allow_pickle=True)
    key = {(c, int(i)): n for n, (c, i) in
           enumerate(zip(z["keys_case"], z["keys_cid"]))}
    p_cf = np.nanmax(z["P_crossfit"], axis=1)
    p_is = np.nanmax(z["P_insample"], axis=1)

    # measured removal outcomes, one row per component
    d_dice, d_qdice, lost, idx = [], [], [], []
    for r in csv.DictReader(open(args.table, newline="", encoding="utf-8")):
        if r["strategy"] != "topology_remove":
            continue
        k = (r["case"], int(r["component_id"]))
        if k not in key:
            continue
        idx.append(key[k])
        d_dice.append(float(r["d_dice"]) if r["d_dice"] not in ("", "nan")
                      else np.nan)
        d_qdice.append(float(r["d_qdice"]) if r["d_qdice"] not in ("", "nan")
                       else np.nan)
        lost.append(float(r["lesions_lost"]) if r["lesions_lost"] not in ("", "nan")
                    else 0.0)
    idx = np.array(idx)
    d_dice = np.array(d_dice)
    d_qdice = np.array(d_qdice)
    lost = np.array(lost)
    print(f"{len(idx)} measured topology_remove actions joined to cross-fitted "
          f"p_failure\n")

    res = {"n_actions": int(len(idx))}
    for tag, p_all in (("insample", p_is), ("crossfit", p_cf)):
        p = p_all[idx]
        print(f"=== {tag} p_failure ===")
        print(f"  {'bin':12s}{'n':>7s}{'mean dDice':>13s}{'lost':>8s}"
              f"{'nan (free)':>12s}")
        rows = []
        for (lo, hi), lab in zip(BINS, BIN_LAB):
            m = (p >= lo) & (p < hi)
            if m.sum() == 0:
                rows.append({"bin": lab, "n": 0})
                continue
            md = float(np.nanmean(d_dice[m])) if np.isfinite(d_dice[m]).any() else float("nan")
            rows.append({"bin": lab, "n": int(m.sum()), "mean_d_dice": md,
                         "lesions_lost": float(lost[m].sum()),
                         "n_tumour_free": int(np.isnan(d_dice[m]).sum())})
            print(f"  {lab:12s}{int(m.sum()):7d}{md:+13.5f}"
                  f"{lost[m].sum():8.0f}{int(np.isnan(d_dice[m]).sum()):12d}")
        res[tag + "_bins"] = rows

        print(f"  {'threshold':12s}{'kept':>7s}{'%':>7s}{'mean dDice':>13s}"
              f"{'lost':>8s}{'precision':>11s}")
        thr = []
        for t in THRESHOLDS:
            m = p >= t
            if m.sum() == 0:
                continue
            v = d_qdice[m]
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            prec = 100 * pos / max(pos + neg, 1)
            thr.append({"threshold": t, "kept": int(m.sum()),
                        "pct": 100 * float(m.mean()),
                        "mean_d_dice": float(np.nanmean(d_dice[m])),
                        "lesions_lost": float(lost[m].sum()),
                        "precision_d_qdice": prec})
            print(f"  >={t:<10.2f}{int(m.sum()):7d}{100*m.mean():7.1f}"
                  f"{np.nanmean(d_dice[m]):+13.5f}{lost[m].sum():8.0f}"
                  f"{prec:10.1f}%")
        res[tag + "_thresholds"] = thr
        print()

    # how many components clear 0.90 at all, per detector (the "34 vs 663")
    res["n_components"] = int(len(p_cf))
    res["n_ge_090_insample"] = int((p_is >= 0.90).sum())
    res["n_ge_090_crossfit"] = int((p_cf >= 0.90).sum())
    res["n_ge_090_unc_insample"] = int((z["unc_insample"] >= 0.90).sum())
    res["n_ge_090_unc_crossfit"] = int(
        (z["unc_crossfit"][np.isfinite(z["unc_crossfit"])] >= 0.90).sum())
    print(f"components with p_failure >= 0.90, of {len(p_cf)}:")
    print(f"  reasoning detector   in-sample {res['n_ge_090_insample']:5d}   "
          f"cross-fitted {res['n_ge_090_crossfit']:5d}")
    print(f"  uncertainty-only     in-sample {res['n_ge_090_unc_insample']:5d}   "
          f"cross-fitted {res['n_ge_090_unc_crossfit']:5d}")

    # matched-rate threshold for the uncertainty detector, cross-fitted
    u = z["unc_crossfit"]
    u = u[np.isfinite(u)]
    target = res["n_ge_090_crossfit"] / len(p_cf)
    res["unc_matched_threshold_crossfit"] = float(np.quantile(u, 1 - target))
    print(f"  matched-rate threshold for uncertainty (cross-fitted): "
          f"{res['unc_matched_threshold_crossfit']:.4f}")

    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
