#!/usr/bin/env python3
"""Is z_rel a lesion prior or a field-of-view artefact?

z_rel is a candidate's centroid height as a fraction of the body extent found
in its own scan. That makes it comparable across patients only if the scans
cover comparable anatomy. AutoPET scans vary: some are head-to-thigh, some stop
at the pelvis, a few include the whole leg. When coverage varies, the same
physical organ lands at different z_rel in different scans, and a detector that
leans on z_rel may be reading the scan protocol rather than the patient.

Three things are reported:

  coverage   the z extent of each scan in mm, and its spread. If every scan
             covered the same anatomy this would be nearly constant.
  profile    positive rate by z_rel decile, so a monotone prior ("lesions sit
             mid-torso") is distinguishable from an edge effect ("everything at
             the very top or bottom of the scan is spurious").
  trimmed    AUROC of z_rel alone over all candidates, and again after dropping
             the top and bottom decile. If the signal survives trimming it is
             about position within the body; if it collapses, it was the edges.

    python scripts/zrel_probe.py --cand reports/cand_cv_oof.csv --kind recover
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

MAPS = {"cand_cv_oof.csv": Path(r"D:\data\cv\oof"),
        "cand_cv_val.csv": Path(r"D:\data\cv\val"),
        "cand_cv_test.csv": Path(r"D:\data\cv\test")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--maps", type=Path, default=None)
    args = ap.parse_args()
    maps = args.maps or MAPS[args.cand.name]

    rows = list(csv.DictReader(open(args.cand, newline="", encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})

    # --- scan coverage, from the map headers alone (no image loaded) ----
    cov = {}
    for c in cases:
        z = np.load(maps / f"{c}.npz")
        shape = tuple(int(v) for v in z["shape"])
        sp = tuple(float(v) for v in z["spacing"])
        cov[c] = shape[0] * sp[0]
    v = np.array(list(cov.values()))
    print(f"scan z coverage over {len(cov)} scans, mm")
    print(f"  min {v.min():7.1f}  p25 {np.percentile(v,25):7.1f}  median "
          f"{np.median(v):7.1f}  p75 {np.percentile(v,75):7.1f}  max {v.max():7.1f}")
    print(f"  mean {v.mean():.1f} +- {v.std():.1f} sd  "
          f"({100*v.std()/v.mean():.1f}% of the mean)")
    print(f"  ratio max/min {v.max()/v.min():.2f}x")

    for kind, pos in (("pred", "FP"), ("recover", "recoverable")):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        y = np.array([1 if r["status"] == pos else 0 for r in sub])
        z = np.array([float(r["z_rel"]) for r in sub])
        print(f"\n--- {kind}: {len(sub)} candidates, {y.sum()} {pos} "
              f"({y.mean():.1%}) ---")
        print(f"z_rel  {pos:>12s}      other   rate    n")
        edges = np.percentile(z, np.arange(0, 101, 10))
        edges[-1] += 1e-9
        for i in range(10):
            m = (z >= edges[i]) & (z < edges[i + 1])
            if not m.any():
                continue
            npos, n = int(y[m].sum()), int(m.sum())
            bar = "#" * int(round(40 * npos / max(n, 1)))
            print(f"{edges[i]:.2f}-{edges[i+1]:.2f} {npos:>6d} {n-npos:>10d} "
                  f"{npos/n:>6.1%} {n:>5d}  {bar}")
        a_all = roc_auc_score(y, z) if len(set(y)) > 1 else float("nan")
        # z_rel alone, and with the extreme deciles removed
        keep = (z > edges[1]) & (z < edges[9])
        a_trim = (roc_auc_score(y[keep], z[keep])
                  if keep.any() and len(set(y[keep])) > 1 else float("nan"))
        print(f"z_rel alone: AUROC {a_all:.4f} over all {len(y)} candidates")
        print(f"             AUROC {a_trim:.4f} over the middle 8 deciles "
              f"({int(keep.sum())} candidates), "
              f"change {a_trim - a_all:+.4f}")
        # and the direction, stated rather than left to the reader
        mp, mn = z[y == 1].mean(), z[y == 0].mean()
        print(f"             mean z_rel {mp:.3f} when {pos}, {mn:.3f} otherwise "
              f"(difference {mp-mn:+.3f})")


if __name__ == "__main__":
    main()
