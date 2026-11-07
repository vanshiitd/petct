#!/usr/bin/env python3
"""Phase XVIII endpoint S (patient harm vs hand-built) and fairness subgroups.

See `reports/phase18_psma_plan.md`. Endpoints D and L and the secondary
endpoints come from `phase13_eval.py --split psma`. A patient is harmed if,
against the hand-built run, they lose a lesion hit or, when tumour-positive,
their Dice drops by more than 0.01. H0: R > DELTA, exact one-sided binomial.

Writes phase18_eval_psma_safety.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read

R = Path(r"D:\data\reports")
TSV = Path(r"D:\data\psma_nifti\PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv")
DELTA, ALPHA = 0.01, 0.05


def main() -> None:
    base = read(R / "p18_psma_hb", "base")
    hb = read(R / "p18_psma_hb", "final")
    a = read(R / "p18_psma_ltt", "final")
    assert set(a) == set(hb) == set(base)
    cases = sorted(a)
    harmed = {c: bool(a[c]["n_hit"] < hb[c]["n_hit"]
                      or (base[c]["n_gt"] > 0 and a[c]["dice"] - hb[c]["dice"] < -0.01))
              for c in cases}
    n, k = len(cases), sum(harmed.values())
    p = float(binom.cdf(k, n, DELTA))
    ub = 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))
    changed = [c for c in cases if any(a[c][m] != hb[c][m] for m in
                                       ("dice", "fp_ml", "fn_ml", "n_hit", "n_fp_comp"))]
    print(f"S: harmed {k}/{n}  p(H0: R > {DELTA}) = {p:.4f}  -> "
          f"{'REJECT (safe)' if p < ALPHA else 'not rejected'}  | exact 95% UB {ub:.4f}")
    print(f"changed cases vs hand-built: {len(changed)}")

    clin = {}
    for r in csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"):
        clin[(r["case_identifier_number"], r["study_date"].replace("-", ""))] = r

    def info(c):
        pid, date, _ = c.rsplit("_", 2)
        return clin.get((pid, date))

    ages = np.array([float(info(c)["Age_at_Imaging"]) for c in cases if info(c)])
    t1, t2 = np.percentile(ages, [100 / 3, 200 / 3])
    groups = {}
    for c in cases:
        m = info(c)
        if m is None:
            continue
        age = float(m["Age_at_Imaging"])
        for g in (f"tracer={m['pet_radionuclide']}", f"vendor={m['Manufacturer']}",
                  f"age={'low' if age <= t1 else 'mid' if age <= t2 else 'high'}"):
            groups.setdefault(g, []).append(c)
    sub = {}
    for g, cs in sorted(groups.items()):
        pos = [c for c in cs if base[c]["n_gt"] > 0]
        d = [a[c]["dice"] - hb[c]["dice"] for c in pos]
        dm = [hb[c]["dice"] - base[c]["dice"] for c in pos]
        sub[g] = {"n": len(cs), "harmed": sum(harmed[c] for c in cs),
                  "dDice_vs_hb": float(np.nanmean(d)) if d else float("nan"),
                  "hb_dDice_vs_M0": float(np.nanmean(dm)) if dm else float("nan"),
                  "M0_dice": float(np.nanmean([base[c]["dice"] for c in pos])) if pos
                  else float("nan")}
        print(f"  {g:28s} n={len(cs):3d} harmed={sub[g]['harmed']} "
              f"M0 Dice={sub[g]['M0_dice']:.3f} HB-M0={sub[g]['hb_dDice_vs_M0']:+.4f} "
              f"P17-HB={sub[g]['dDice_vs_hb']:+.5f}")
    (R / "phase18_eval_psma_safety.json").write_text(json.dumps({
        "delta": DELTA, "alpha": ALPHA, "n": n, "harmed": k, "p": p, "ub95": ub,
        "safe": p < ALPHA, "harmed_cases": [c for c in cases if harmed[c]],
        "changed_cases": changed, "age_tertiles": [t1, t2], "subgroups": sub},
        indent=1, default=float))


if __name__ == "__main__":
    main()
