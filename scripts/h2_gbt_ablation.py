#!/usr/bin/env python3
"""H2 reanalysis: the fair GBT ablation, removal task only.

`exp3_detect.py` reports one GBT, fit on the full ('d all') feature set, and
contrasts it against the *logistic* 'd all' - never against uncertainty
alone fit with the same architecture. The deployed detector
(`phase4_diag_model.py`) is a HistGradientBoostingClassifier, so the
contrast that actually tests whether reasoning features help the model that
is really used has never been run. This fits a GBT on 'b unc' as well as
'd all' (hyperparameters from `exp3_detect.tune_gbt`, grouped CV on the
fitting split alone, same grid), and bootstraps the one new contrast on val
and test with the same `paired_bootstrap` function used for every other H2
number -- re-using `exp3_detect.py`'s library code unchanged, not
re-implementing it.

Pre-registered in `reports/h2h4_reanalysis_plan.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, GROUPS, tune_gbt, paired_bootstrap

R = Path(r"D:\data\reports")
DROP = ["z_rel"]
ADD = ["dz_liver_mm", "dz_heart_mm", "dz_bladder_mm", "dz_kidney_mm",
       "edge_body_mm", "edge_scan_mm"]


def apply_ablation() -> dict:
    groups = {k: [f for f in v if f not in DROP] for k, v in GROUPS.items()}
    for k in ("c reason", "d all"):
        groups[k] = groups[k] + [f for f in ADD if f not in groups[k]]
    return groups


def fit_gbt(name: str, groups: dict, rows_v, yv, gv, med_cache: dict):
    feats = groups[name]
    Xv, med, _ = matrix(rows_v, feats, None)
    med_cache[name] = med
    best, table = tune_gbt(Xv, yv, gv)
    print(f"  GBT({name}): best CV AUROC {table[0][0]:.4f}  params {best}  "
          f"(grid spread {table[0][0] - table[-1][0]:.4f})")
    m = HistGradientBoostingClassifier(random_state=0, early_stopping=False,
                                        **best)
    m.fit(Xv, yv)
    return m, med


def main() -> None:
    groups = apply_ablation()
    print(f"'d all' = {len(groups['d all'])} features (z_rel dropped, "
          f"6 anatomical features added)")

    kind = "pred"  # removal task -- H2's negative result is specific to this
    rows_f, yf, gf, _ = load(R / "cand_cv_oof_pos.csv", kind)
    rows_v, yv_s, _, _ = load(R / "cand_cv_val_pos.csv", kind)
    rows_t, yt_s, _, _ = load(R / "cand_cv_test_pos.csv", kind)
    print(f"\nfit {len(yf)} candidates ({yf.sum()} positive, {yf.mean():.1%})  "
          f"val {len(yv_s)} ({yv_s.sum()} positive)  "
          f"test {len(yt_s)} ({yt_s.sum()} positive)")

    med_cache: dict = {}
    scores = {"val": {}, "test": {}}
    for name in ("b unc", "d all"):
        m, med = fit_gbt(name, groups, rows_f, yf, gf, med_cache)
        feats = groups[name]
        Xv, _, _ = matrix(rows_v, feats, med)
        Xt, _, _ = matrix(rows_t, feats, med)
        scores["val"][name] = m.predict_proba(Xv)[:, 1]
        scores["test"][name] = m.predict_proba(Xt)[:, 1]
        print(f"    val AUROC {roc_auc_score(yv_s, scores['val'][name]):.4f}  "
              f"test AUROC {roc_auc_score(yt_s, scores['test'][name]):.4f}")

    out = {}
    for split, y in (("val", yv_s), ("test", yt_s)):
        d, lo, hi = paired_bootstrap(y, scores[split]["b unc"],
                                      scores[split]["d all"])
        excl = (lo > 0) == (hi > 0)
        print(f"\npaired bootstrap on {split}, (d all GBT) - (b unc GBT): "
              f"{d:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] "
              f"{'excludes 0' if excl else 'straddles 0'}")
        out[split] = {"delta_auroc": d, "ci_lo": lo, "ci_hi": hi,
                       "excludes_zero": bool(excl),
                       "auroc_b_unc": float(roc_auc_score(y, scores[split]["b unc"])),
                       "auroc_d_all": float(roc_auc_score(y, scores[split]["d all"]))}

    print("\nreference (from the original, logistic-model H2 report):")
    print("  test, (d all) - (b unc), logistic: -0.0106 [-0.0289, +0.0103] straddles 0")

    passes = out["test"]["excludes_zero"] and out["test"]["delta_auroc"] > 0
    verdict = ("H2 removal: SUPPORTED with the GBT architecture" if passes else
               "H2 removal: still not supported, even with GBT")
    print(f"\nVerdict: {verdict}")
    out["verdict"] = verdict
    out_path = R / "h2_gbt_ablation.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
