#!/usr/bin/env python3
"""Score the refinement network (strategy 7) with the agent's own metrics.

The refined masks come out of `nnUNetv2_predict` in the geometry of the
Dataset505 images, which `phase4_refine_inputs.py` already checked is the
geometry of the probability maps, so a refined mask can simply be dropped into
a CaseContext in place of the M0 components and scored by `ctx.score()`. Using
the same function matters: Dice, FP volume and lesion hits here are then the
identical definitions used for M0 and for the agent, and the three rows are
comparable without any further adjustment.

Two comparisons are reported because they answer different questions:

  vs the ensemble   what a deployed system would see, but handicapped -- the
                    refinement network is one model and M0 is five, and Phase
                    IIIb measured that gap at +0.0213 Dice on its own
  vs single model   the matched comparison, one model against one model

The output CSV uses the same columns as the agent's `_final.csv`, so the
ablation ladder can pick it up as one more variant.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from evaluate_predictions import hd95_mm

COLS = ["case", "dice", "fn_ml", "fp_ml", "hd95", "n_fp_comp", "n_gt", "n_hit",
        "pred_ml"]


def inject(ctx: CaseContext, mask: np.ndarray) -> None:
    """Put an arbitrary mask into the context as the current prediction."""
    lab, n = ndi.label(mask)
    ctx.labels = lab
    ctx.small = np.zeros_like(mask, bool)
    ctx.comp_ids = sorted(int(c) for c in np.unique(lab) if c)
    ctx._slices = {c: s for c, s in zip(range(1, n + 1), ndi.find_objects(lab))
                   if s is not None}
    ctx.next_id = n + 1
    ctx._rec_cache = None


def score(ctx: CaseContext) -> dict:
    s = ctx.score()
    m = ctx.mask()
    s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                 if m.any() and ctx.gt.any() else float("nan"))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True,
                    help="directory of refined masks from nnUNetv2_predict")
    ap.add_argument("--maps", type=Path, required=True,
                    help="probability maps the baseline is read from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.pred.glob("*.nii.gz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} refined masks from {args.pred}", flush=True)

    base, ref = [], []
    for i, f in enumerate(files, 1):
        case = f.name[: -len(".nii.gz")]
        ctx = CaseContext(case, args.maps)
        b = score(ctx)
        b["case"] = case
        base.append(b)
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
        if m.shape != ctx.shape:
            raise SystemExit(f"{case}: refined {m.shape} != map {ctx.shape}")
        inject(ctx, m)
        r = score(ctx)
        r["case"] = case
        ref.append(r)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("base", base), ("final", ref)):
        p = args.out.with_name(args.out.name + f"_{name}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in COLS} for r in rows])
    print(f"written {args.out}_base.csv and _final.csv")

    pb = [r for r in base if r["n_gt"] > 0]
    pr = [r for r in ref if r["n_gt"] > 0]
    ngt = sum(r["n_gt"] for r in pb)

    def mn(rs, k):
        v = [r[k] for r in rs if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{len(pb)} tumour-positive of {len(base)} cases, "
          f"{int(ngt)} ground-truth lesions")
    print(f"{'':10s}{'Dice':>9s}{'FP mL':>9s}{'FN mL':>9s}{'HD95':>9s}"
          f"{'lesSens':>9s}")
    # The plan defines Dice and HD95 over tumour-positive cases but FP and FN
    # volume as means over ALL cases, so the two groups are not interchangeable
    # here: reporting FP over positives only would leave out every tumour-free
    # patient, which is exactly where a false positive matters most.
    for nm, rs, al in (("baseline", pb, base), ("refined", pr, ref)):
        print(f"{nm:10s}{mn(rs,'dice'):9.4f}{mn(al,'fp_ml'):9.3f}"
              f"{mn(al,'fn_ml'):9.3f}{mn(rs,'hd95'):9.2f}"
              f"{sum(r['n_hit'] for r in rs)/ngt:9.4f}")
    print(f"{'delta':10s}{mn(pr,'dice')-mn(pb,'dice'):+9.4f}"
          f"{mn(ref,'fp_ml')-mn(base,'fp_ml'):+9.3f}"
          f"{mn(ref,'fn_ml')-mn(base,'fn_ml'):+9.3f}"
          f"{mn(pr,'hd95')-mn(pb,'hd95'):+9.2f}"
          f"{(sum(r['n_hit'] for r in pr)-sum(r['n_hit'] for r in pb))/ngt:+9.4f}")
    nneg = [r for r in base if r["n_gt"] == 0]
    fneg = {r["case"]: r for r in ref}
    print(f"tumour-free patients with any FP: "
          f"{sum(1 for r in nneg if r['fp_ml'] > 0)} -> "
          f"{sum(1 for r in nneg if fneg[r['case']]['fp_ml'] > 0)} of {len(nneg)}")


if __name__ == "__main__":
    main()
