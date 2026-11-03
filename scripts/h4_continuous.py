#!/usr/bin/env python3
"""H4 reanalysis: continuous difficulty, not tertile extremes.

`final_eval_tables.py` section 22 bins Q and u_stab into tertiles and tests
only the hardest bin against the easiest -- 29 of 353 cases for Q, 48 of 353
for u_stab. That is most of the sample thrown away. This re-uses the exact
same per-case data (`fe_oof_finalagent_traces.csv` for Q, `final_case_groups.json`
for `u_stab`, `fe_oof_reason_base.csv` / `fe_oof_finalagent_final.csv` for
per-case Dice before/after) and asks the monotonic-association question
directly: is dDice correlated with difficulty across all tumour-positive
out-of-fold cases, not just the hardest third of them.

Two statistics, both bootstrapped over the same case list the rest of this
project's dDice intervals already resample (`phase5_eval.paired_dice`'s own
convention, 2000 resamples):

  - Spearman rho between a "harder = larger value" difficulty score and the
    per-case dDice -- the natural statistic for "gains are larger in harder
    cases" without assuming a linear relationship;
  - the OLS slope of dDice on the difficulty score's percentile rank, for a
    directly interpretable effect size (dDice gained per decile of
    difficulty).

Pre-registered in `reports/h2h4_reanalysis_plan.md`. Validation and test are
read as a check, exactly as section 22 already did -- no new touch of
either split beyond what the original report already made.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, NBOOT

R = Path(r"D:\data\reports")


def per_case_delta(sp: str) -> dict:
    base = read(R / f"fe_{sp}_reason", "base")
    fin = read(R / f"fe_{sp}_finalagent", "final")
    return {c: fin[c]["dice"] - base[c]["dice"] for c in base
            if c in fin and base[c]["n_gt"] > 0}


def difficulty_scores(sp: str, diag: dict) -> dict:
    tr = read(R / f"fe_{sp}_finalagent", "traces")
    q = {c: tr[c]["q0"] for c in tr}          # low Q = hard
    u = {c: diag[sp][c]["ustab"] for c in diag[sp]}  # high u_stab = hard
    return {"Q": {c: -v for c, v in q.items() if np.isfinite(v)},  # flip: high = hard
            "u_stab": {c: v for c, v in u.items() if np.isfinite(v)}}


def bootstrap_stat(delta: np.ndarray, score: np.ndarray, stat_fn, seed=2026):
    n = len(delta)
    rng = np.random.default_rng(seed)
    base = stat_fn(delta, score)
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, n, n)
        boots[i] = stat_fn(delta[ix], score[ix])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "value": float(base), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool((lo > 0) == (hi > 0))}


def spearman_stat(delta, score):
    from scipy.stats import spearmanr
    r = spearmanr(score, delta).statistic
    return 0.0 if not np.isfinite(r) else r


def ols_slope_stat(delta, score):
    rank = (np.argsort(np.argsort(score)) + 1) / len(score)  # percentile rank
    rank = rank - rank.mean()
    d = delta - delta.mean()
    denom = float((rank ** 2).sum())
    return float((rank * d).sum() / denom) if denom else 0.0


def main() -> None:
    diag = json.load(open(R / "final_case_groups.json", encoding="utf-8"))
    results = {}
    for sp in ("oof", "val", "test"):
        delta_map = per_case_delta(sp)
        scores = difficulty_scores(sp, diag)
        print(f"\n{'='*78}\n{sp.upper()}{'  (descriptive)' if sp == 'test' else ''}")
        results[sp] = {}
        for name, score_map in scores.items():
            common = sorted(set(delta_map) & set(score_map))
            if len(common) < 10:
                print(f"  {name:10s} n={len(common)}  too few")
                continue
            delta = np.array([delta_map[c] for c in common])
            score = np.array([score_map[c] for c in common])
            rho = bootstrap_stat(delta, score, spearman_stat)
            slope = bootstrap_stat(delta, score, ols_slope_stat)
            print(f"  {name:10s} n={rho['n']:4d}  "
                  f"Spearman rho(harder,dDice) {rho['value']:+.4f} "
                  f"[{rho['lo']:+.4f}, {rho['hi']:+.4f}] "
                  f"{'excludes 0' if rho['excludes_zero'] else 'straddles 0'}  |  "
                  f"slope/decile {slope['value']/10:+.5f} "
                  f"[{slope['lo']/10:+.5f}, {slope['hi']/10:+.5f}]")
            results[sp][name] = {"spearman": rho, "slope_per_decile":
                                  {**slope, "value": slope["value"] / 10,
                                   "lo": slope["lo"] / 10, "hi": slope["hi"] / 10}}

    print(f"\n{'='*78}")
    print("reference (tertile test, final_eval.md section 2):")
    print("  Q low (hardest) n=29   dDice +0.0209 [+0.0003, +0.0512]")
    print("  u_stab high (hardest) n=48  dDice +0.0162 [+0.0004, +0.0451]")

    oof = results["oof"]
    passes = all(oof[k]["spearman"]["excludes_zero"] and oof[k]["spearman"]["value"] > 0
                 for k in ("Q", "u_stab") if k in oof)
    verdict = ("H4: supported, more precisely, by the continuous test" if passes
               else "H4: continuous test does not improve on the tertile result")
    print(f"\nVerdict: {verdict}")

    out_path = R / "h4_continuous.json"
    out_path.write_text(json.dumps({"results": results, "verdict": verdict},
                                    indent=1, default=float), encoding="utf-8")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
