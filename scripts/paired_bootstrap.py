#!/usr/bin/env python3
"""Paired bootstrap over patients for the difference between two runs.

Both runs are scored on the same patients, so the comparison is paired: the
per-patient differences are resampled, not the two score sets independently.
That removes between-patient variance, which on this dataset dwarfs the
between-model difference (per-case Dice ranges from 0 to ~0.95).

Reports, for the chosen metric:
  * each run's mean, and the paired mean difference B - A
  * a percentile bootstrap 95% CI for that difference over `--resamples` draws
  * per-patient wins/losses/ties, and a two-sided sign-flip p-value

A CI that straddles zero means this test set cannot tell the two apart at this
sample size -- not that the models are equal.

    python scripts/paired_bootstrap.py --a-label pretrained --b-label scratch \\
        --a reports/eval_505_test.csv --b reports/eval_505s_test.csv \\
        --metric dice --positives-only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path, metric: str, positives_only: bool) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if positives_only:
        rows = [r for r in rows if r["gt_positive"] == "1"]
    return {r["case"]: float(r[metric]) for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--a-label", default="A")
    p.add_argument("--b-label", default="B")
    p.add_argument("--metric", default="dice")
    p.add_argument("--positives-only", action="store_true",
                   help="restrict to tumour-positive cases (right for Dice)")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--higher-is-better", dest="hib", action="store_true", default=True)
    p.add_argument("--lower-is-better", dest="hib", action="store_false")
    args = p.parse_args()

    a_map = load(args.a, args.metric, args.positives_only)
    b_map = load(args.b, args.metric, args.positives_only)
    cases = sorted(set(a_map) & set(b_map))
    if len(cases) != len(a_map) or len(cases) != len(b_map):
        print(f"WARNING: pairing on {len(cases)} of {len(a_map)}/{len(b_map)} cases")
    a = np.array([a_map[c] for c in cases])
    b = np.array([b_map[c] for c in cases])
    d = b - a

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(cases), size=(args.resamples, len(cases)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    better = d > 0 if args.hib else d < 0
    worse = d < 0 if args.hib else d > 0
    nb, nw = int(better.sum()), int(worse.sum())
    # two-sided sign test over the cases that are not ties
    n = nb + nw
    if n:
        null = (rng.random((args.resamples, n)) < 0.5).sum(axis=1)
        pval = float((np.minimum(null, n - null) <= min(nb, nw)).mean())
    else:
        pval = float("nan")

    direction = "higher is better" if args.hib else "lower is better"
    print(f"\nmetric: {args.metric}  ({direction})")
    print(f"patients paired: {len(cases)}"
          f"{'  (tumour-positive only)' if args.positives_only else ''}")
    print(f"resamples: {args.resamples}, seed {args.seed}")
    print("-" * 64)
    print(f"{args.a_label:>22s} mean : {a.mean():.4f}")
    print(f"{args.b_label:>22s} mean : {b.mean():.4f}")
    print(f"{'paired difference':>22s}      : {d.mean():+.4f}  "
          f"({args.b_label} - {args.a_label})")
    print(f"{'95% CI':>22s}      : [{lo:+.4f}, {hi:+.4f}]  "
          f"{'excludes 0' if (lo > 0) == (hi > 0) else 'straddles 0'}")
    print(f"{'median difference':>22s}      : {np.median(d):+.4f}")
    print("-" * 64)
    print(f"{args.b_label} better on {nb}/{len(cases)} patients, "
          f"worse on {nw}, tied on {len(cases) - nb - nw}")
    print(f"sign test p = {pval:.4f}")
    if d.size:
        k = min(5, d.size)
        order = np.argsort(d)
        print(f"\nlargest swings ({args.b_label} - {args.a_label}):")
        for i in list(order[:k]) + list(order[-k:][::-1]):
            print(f"   {cases[i]:34s} {a[i]:7.4f} -> {b[i]:7.4f}   {d[i]:+.4f}")


if __name__ == "__main__":
    main()
