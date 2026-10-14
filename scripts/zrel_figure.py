#!/usr/bin/env python3
r"""Figure: z_rel for positive vs negative candidates, and what replaces it.

Four panels:
  (a) scan z coverage across the cohort -- the reason z_rel is not comparable
      between patients
  (b) z_rel distribution, recoverable vs not
  (c) z_rel distribution, false positive vs true positive
  (d) positive rate by z_rel decile for both tasks, which is where the
      non-monotone edge effect shows up that a histogram hides

Drawn from the candidate CSVs, so the figure and the numbers in the report come
from one source.

    python scripts/zrel_figure.py --oof reports/cand_cv_oof.csv \
        --test reports/cand_cv_test.csv --out reports/fig_zrel.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAPS = {"oof": Path(r"D:\data\cv\oof"), "test": Path(r"D:\data\cv\test")}


def read(path: Path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = {"oof": read(args.oof), "test": read(args.test)}

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) coverage
    for tag, c in (("oof", "#3b6ea5"), ("test", "#c1673b")):
        cases = sorted({r["case"] for r in rows[tag]})
        cov = []
        for cs in cases:
            z = np.load(MAPS[tag] / f"{cs}.npz")
            cov.append(int(z["shape"][0]) * float(z["spacing"][0]))
        ax[0, 0].hist(cov, bins=30, alpha=0.6, color=c,
                      label=f"{tag} (n={len(cov)}, {min(cov):.0f}-{max(cov):.0f} mm)")
    ax[0, 0].set_xlabel("scan z coverage (mm)")
    ax[0, 0].set_ylabel("scans")
    ax[0, 0].set_title("(a) coverage varies 3.3x, so z_rel is not\n"
                       "comparable between patients", fontsize=10)
    ax[0, 0].legend(fontsize=8)

    # (b) and (c) distributions
    for j, (kind, pos, lab) in enumerate((("recover", "recoverable", "recoverable"),
                                          ("pred", "FP", "false positive"))):
        a = ax[0, 1] if j == 0 else ax[1, 0]
        sub = [r for r in rows["oof"] if r["kind"] == kind]
        z = np.array([float(r["z_rel"]) for r in sub])
        y = np.array([r["status"] == pos for r in sub])
        bins = np.linspace(0, 1, 26)
        a.hist(z[y], bins=bins, alpha=0.62, color="#b2402f", density=True,
               label=f"{lab} (n={int(y.sum())})")
        a.hist(z[~y], bins=bins, alpha=0.62, color="#4a7ba6", density=True,
               label=f"not {lab} (n={int((~y).sum())})")
        a.set_xlabel("z_rel (0 = inferior edge of body, 1 = superior edge)")
        a.set_ylabel("density")
        a.set_title(f"({'bc'[j]}) {kind} task, out-of-fold", fontsize=10)
        a.legend(fontsize=8)

    # (d) positive rate by decile -- the edge effect
    for kind, pos, c, m in (("pred", "FP", "#b2402f", "o"),
                            ("recover", "recoverable", "#2f6b4f", "s")):
        for tag, ls in (("oof", "-"), ("test", "--")):
            sub = [r for r in rows[tag] if r["kind"] == kind]
            z = np.array([float(r["z_rel"]) for r in sub])
            y = np.array([r["status"] == pos for r in sub], dtype=float)
            e = np.percentile(z, np.arange(0, 101, 10)); e[-1] += 1e-9
            xs, ys = [], []
            for i in range(10):
                mk = (z >= e[i]) & (z < e[i + 1])
                if mk.any():
                    xs.append(0.5 * (e[i] + e[i + 1])); ys.append(y[mk].mean())
            ax[1, 1].plot(xs, ys, ls, color=c, marker=m, ms=4,
                          label=f"{kind}, {tag}")
    ax[1, 1].set_xlabel("z_rel decile midpoint")
    ax[1, 1].set_ylabel("positive rate")
    ax[1, 1].set_title("(d) flat in the middle, extreme at the edges:\n"
                       "an edge indicator, not an anatomical prior", fontsize=10)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
