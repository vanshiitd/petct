#!/usr/bin/env python3
"""Audit R3: the dumb baseline the agent should be measured against.

Remove small predicted components from M0 and see how much of the agent's
benefit a volume threshold reproduces. Two rules:

  fixed     the Phase I/II post-processing rule, fixed on single-model
            validation: drop components under 0.5 mL, or with at least 0.10 of
            their volume in physiological uptake / liver / spleen
  tuned     a minimum volume re-tuned on M0 out-of-fold **only**: the largest
            threshold whose lesion-sensitivity cost stays within -0.01, which is
            the same safety margin the agent is held to. Checked on validation,
            then applied unchanged to test.

Scoring goes through the same `CaseContext` the agent uses, so these rows and
the agent's rows come from one metric implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from evaluate_predictions import hd95_mm

GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
ORGAN_FRAC = 0.10
FIXED_MIN_ML = 0.5


def score(ctx: CaseContext) -> dict:
    s = ctx.score()
    m = ctx.mask()
    s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                 if m.any() and ctx.gt.any() else float("nan"))
    return s


def apply_rule(ctx: CaseContext, min_ml: float, use_organ: bool) -> None:
    """Delete every component below the volume floor, in place."""
    for cid in list(ctx.comp_ids):
        sl = ctx.comp_slice(cid)
        f = ctx.features(sl, ctx.labels[sl] == cid)
        if f is None:
            continue
        drop = f["vol_ml"] < min_ml
        if use_organ and not drop:
            organ = (f.get("frac_physio", 0.0) + f.get("frac_liver", 0.0)
                     + f.get("frac_spleen", 0.0))
            drop = organ >= ORGAN_FRAC
        if drop:
            loc = ctx.labels[sl]
            loc[loc == cid] = 0
            ctx.labels[sl] = loc


KEYS = ("dice", "fp_ml", "fn_ml", "hd95", "n_fp_comp", "n_gt", "n_hit",
        "pred_ml")


def _score(ctx: CaseContext, with_hd95: bool) -> dict:
    """HD95 is by far the slowest metric and the tuner does not use it."""
    s = ctx.score()
    if with_hd95:
        m = ctx.mask()
        s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                     if m.any() and ctx.gt.any() else float("nan"))
    else:
        s["hd95"] = float("nan")
    return s


def run_split(maps: Path, min_ml: float, use_organ: bool, limit=None,
              with_hd95: bool = True):
    cases = sorted(f.stem for f in maps.glob("*.npz"))
    if limit:
        cases = cases[:limit]
    base, final = [], []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        ctx = CaseContext(c, maps)
        b = _score(ctx, with_hd95)
        apply_rule(ctx, min_ml, use_organ)
        f = _score(ctx, with_hd95)
        base.append({"case": c, **{k: b[k] for k in KEYS}})
        final.append({"case": c, **{k: f[k] for k in KEYS}})
        if i % 100 == 0:
            print(f"    {i}/{len(cases)} {(time.time()-t0)/i:.2f}s/case",
                  flush=True)
    return base, final


def run_grid(maps: Path, grid, use_organ: bool, limit=None):
    """Every threshold from one pass over the data.

    Loading a whole-body case and labelling its components costs far more than
    applying a volume floor to it, so the grid is swept inside the per-case
    loop rather than by re-reading the split ten times.
    """
    cases = sorted(f.stem for f in maps.glob("*.npz"))
    if limit:
        cases = cases[:limit]
    base, finals = [], {t: [] for t in grid}
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        ctx = CaseContext(c, maps)
        b = _score(ctx, False)
        base.append({"case": c, **{k: b[k] for k in KEYS}})
        keep = ctx.labels.copy()
        for t in grid:
            ctx.labels = keep.copy()
            apply_rule(ctx, t, use_organ)
            f = _score(ctx, False)
            finals[t].append({"case": c, **{k: f[k] for k in KEYS}})
        ctx.labels = keep
        if i % 100 == 0:
            print(f"    {i}/{len(cases)} {(time.time()-t0)/i:.2f}s/case",
                  flush=True)
    return base, finals


def agg(base, final) -> dict:
    pos = [i for i, r in enumerate(base) if r["n_gt"] > 0]
    neg = [i for i, r in enumerate(base) if r["n_gt"] == 0]
    hit, gt = sum(r["n_hit"] for r in final), sum(r["n_gt"] for r in final)
    hit0 = sum(r["n_hit"] for r in base)
    return {
        "n": len(base), "n_pos": len(pos), "n_neg": len(neg),
        "dice": float(np.nanmean([final[i]["dice"] for i in pos])),
        "dice_base": float(np.nanmean([base[i]["dice"] for i in pos])),
        "neg_with_fp": int(sum(1 for i in neg if final[i]["n_fp_comp"] > 0)),
        "neg_with_fp_base": int(sum(1 for i in neg if base[i]["n_fp_comp"] > 0)),
        "fp_ml": float(np.nanmean([r["fp_ml"] for r in final])),
        "fp_ml_base": float(np.nanmean([r["fp_ml"] for r in base])),
        "fn_ml": float(np.nanmean([r["fn_ml"] for r in final])),
        "hd95": float(np.nanmean([final[i]["hd95"] for i in pos])),
        "lesion_sens": hit / gt if gt else float("nan"),
        "d_lesion_sens": (hit - hit0) / gt if gt else float("nan"),
    }


def write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "apply"), required=True)
    ap.add_argument("--min-ml", type=float, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_baseline.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    M = {"oof": Path(r"D:\data\cv\oof"), "val": Path(r"D:\data\cv\val"),
         "test": Path(r"D:\data\cv\test")}
    R = Path(r"D:\data\reports")

    if args.stage == "tune":
        # Out-of-fold only. Largest threshold whose lesion-sensitivity cost
        # stays within -0.01; test is not read here.
        res = []
        base, finals = run_grid(M["oof"], GRID, False, args.limit)
        for t in GRID:
            a = agg(base, finals[t])
            a["min_ml"] = t
            res.append(a)
            print(f"  min {t:4.2f} mL  Dice {a['dice']:.4f}  "
                  f"negFP {a['neg_with_fp']}/{a['n_neg']}  "
                  f"FP {a['fp_ml']:.3f}  dSens {a['d_lesion_sens']:+.4f}",
                  flush=True)
        ok = [a for a in res if a["d_lesion_sens"] >= -0.01]
        best = max(ok, key=lambda a: a["min_ml"]) if ok else res[0]
        print(f"\nchosen min volume {best['min_ml']} mL "
              f"(largest with dSens >= -0.01 on out-of-fold)")
        args.out.write_text(json.dumps(
            {"grid": res, "chosen_min_ml": best["min_ml"]}, indent=1),
            encoding="utf-8")
        return

    out = {}
    for rule, min_ml, organ in (("fixed", FIXED_MIN_ML, True),
                                ("tuned", args.min_ml, False)):
        out[rule] = {}
        for sp in ("oof", "val", "test"):
            print(f"\n{rule} rule, {sp}:")
            b, f = run_split(M[sp], min_ml, organ, args.limit)
            out[rule][sp] = agg(b, f)
            write(f, R / f"audit_baseline_{rule}_{sp}_final.csv")
            write(b, R / f"audit_baseline_{rule}_{sp}_base.csv")
            a = out[rule][sp]
            print(f"  Dice {a['dice_base']:.4f} -> {a['dice']:.4f}  "
                  f"negFP {a['neg_with_fp_base']}->{a['neg_with_fp']}"
                  f"/{a['n_neg']}  FP {a['fp_ml_base']:.3f}->{a['fp_ml']:.3f}  "
                  f"dSens {a['d_lesion_sens']:+.4f}")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
