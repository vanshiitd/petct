#!/usr/bin/env python3
"""Audit R6(1): paired bootstrap for component-level AUROC, U_stab vs U_ent.

Phase II reported both AUROCs but never an interval on their difference, so
"U_stab wins at component level" has never had an error bar. Resampling is over
**patients**, not components, because components within a patient are not
independent: a case with a scanner artefact contributes many correlated rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from q_model import patient_of

NBOOT = 2000


def auroc_contrast(cand: Path, label: str, seed: int = 0) -> dict:
    rows, y, _, _ = load(cand, "pred")
    pat = np.array([patient_of(r["case"]) for r in rows])
    X, _, _ = matrix(rows, ["ustab_mean", "uent_mean"], None)
    us, ue = X[:, 0], X[:, 1]
    # y == 1 marks a false positive, so higher uncertainty should predict it
    a_s, a_e = roc_auc_score(y, us), roc_auc_score(y, ue)

    pats = np.array(sorted(set(pat)))
    idx = {p: np.where(pat == p)[0] for p in pats}
    rng = np.random.default_rng(seed)
    d = np.empty(NBOOT)
    bs, be = np.empty(NBOOT), np.empty(NBOOT)
    n = 0
    for b in range(NBOOT):
        pick = rng.choice(pats, len(pats), replace=True)
        sel = np.concatenate([idx[p] for p in pick])
        yy = y[sel]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        s_, e_ = roc_auc_score(yy, us[sel]), roc_auc_score(yy, ue[sel])
        bs[n], be[n], d[n] = s_, e_, s_ - e_
        n += 1
    d, bs, be = d[:n], bs[:n], be[:n]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    out = {"split": label, "n_components": int(len(y)),
           "n_patients": int(len(pats)), "n_fp": int(y.sum()),
           "auroc_ustab": float(a_s), "auroc_uent": float(a_e),
           "delta": float(a_s - a_e), "lo": float(lo), "hi": float(hi),
           "p": float(min(1.0, max(p, 1.0 / NBOOT))), "nboot_used": n}
    print(f"{label:10s} n={len(y):5d} comps / {len(pats):4d} patients  "
          f"U_stab {a_s:.4f}  U_ent {a_e:.4f}  "
          f"delta {a_s - a_e:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={out['p']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_auroc_boot.json"))
    args = ap.parse_args()
    R = Path(r"D:\data\reports")
    res = [auroc_contrast(R / "cand_cv_test_pos.csv", "test"),
           auroc_contrast(R / "cand_cv_oof_pos.csv", "oof"),
           auroc_contrast(R / "cand_cv_val_pos.csv", "val")]
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
