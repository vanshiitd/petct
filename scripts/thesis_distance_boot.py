#!/usr/bin/env python3
"""Intervals for the distance and connectivity results. DESCRIPTIVE.

Dice was the pre-specified endpoint. ASSD, HD95, IoU and component counts are
**secondary descriptive** results and are labelled that way wherever they
appear: quoting one of them as a headline after seeing the numbers would be the
same post-hoc endpoint substitution the audit criticises elsewhere.

Paired bootstrap over patients, 2000 resamples, seed 2026. Every number comes
from per-case metrics already on disk - the masks saved for R4 and the base-model
scoring - so no agent is re-run and nothing is tuned.

  Phase V agent vs M0, final agent vs M0   dASSD, dHD95, dIoU, dcomponents
  Swin f0 vs nnU-Net f0                    dcomponents, dlesion-count error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from evaluate_predictions import iou, assd_mm, hd95_mm, dice, component_counts
from q_model import patient_of

R = Path(r"D:\data\reports")
MAPS = {"val": Path(r"D:\data\cv\val"), "test": Path(r"D:\data\cv\test")}
NBOOT, SEED = 2000, 2026


def from_masks(mask_dir: Path, maps: Path) -> dict:
    out = {}
    for f in sorted(mask_dir.glob("*.npz")):
        case = f.stem
        p = np.load(f)["mask"].astype(bool)
        ctx = CaseContext(case, maps)
        g, sp = ctx.gt, ctx.spacing
        npred, ngt = component_counts(p, g)
        out[case] = {
            "iou": iou(p, g) if g.any() else np.nan,
            "assd": assd_mm(p, g, sp) if p.any() and g.any() else np.nan,
            "hd95": hd95_mm(p, g, sp) if p.any() and g.any() else np.nan,
            "n_comp": float(npred), "comp_err": float(npred - ngt),
            "gt_positive": bool(g.any()),
        }
    return out


def from_maps(maps: Path) -> dict:
    out = {}
    for f in sorted(maps.glob("*.npz")):
        case = f.stem
        ctx = CaseContext(case, maps)
        p, g, sp = ctx.mask(), ctx.gt, ctx.spacing
        npred, ngt = component_counts(p, g)
        out[case] = {
            "iou": iou(p, g) if g.any() else np.nan,
            "assd": assd_mm(p, g, sp) if p.any() and g.any() else np.nan,
            "hd95": hd95_mm(p, g, sp) if p.any() and g.any() else np.nan,
            "n_comp": float(npred), "comp_err": float(npred - ngt),
            "gt_positive": bool(g.any()),
        }
    return out


def boot(a: dict, b: dict, key: str, pos_only: bool, label: str) -> dict:
    cases = sorted(set(a) & set(b))
    if pos_only:
        cases = [c for c in cases if b[c]["gt_positive"]]
    pats = sorted({patient_of(c) for c in cases})
    by = {}
    for c in cases:
        by.setdefault(patient_of(c), []).append(c)
    va = np.array([a[c][key] for c in cases], float)
    vb = np.array([b[c][key] for c in cases], float)
    delta = float(np.nanmean(va) - np.nanmean(vb))
    rng = np.random.default_rng(SEED)
    bt = np.empty(NBOOT)
    pos = {c: i for i, c in enumerate(cases)}
    for i in range(NBOOT):
        pick = rng.choice(pats, len(pats), replace=True)
        ix = [pos[c] for p in pick for c in by[p]]
        bt[i] = np.nanmean(va[ix]) - np.nanmean(vb[ix])
    lo, hi = np.percentile(bt, [2.5, 97.5])
    print(f"    {label:30s}{np.nanmean(vb):9.3f} ->{np.nanmean(va):9.3f}  "
          f"{delta:+9.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"{'' if lo > 0 or hi < 0 else '   (straddles 0)'}")
    return {"metric": key, "n_cases": len(cases), "n_patients": len(pats),
            "base": float(np.nanmean(vb)), "agent": float(np.nanmean(va)),
            "delta": delta, "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "thesis_distance_boot.json")
    args = ap.parse_args()
    res = {"note": "DESCRIPTIVE secondary results. Dice was the pre-specified "
                   "endpoint; these are not a substitute for it.",
           "nboot": NBOOT, "seed": SEED}

    for split in ("test", "val"):
        print(f"\n{'='*94}\n{split.upper()}  (descriptive)")
        m0 = from_maps(MAPS[split])
        res[split] = {}
        for label, pre in (("Phase V agent", "pv"), ("final agent", "fa")):
            d = Path(r"D:\data\masks") / f"{pre}_{split}"
            if not d.exists():
                print(f"  {label}: masks not found")
                continue
            a = from_masks(d, MAPS[split])
            print(f"  {label} vs M0:")
            res[split][label] = {
                "assd": boot(a, m0, "assd", True, "ASSD mm (tumour-positive)"),
                "hd95": boot(a, m0, "hd95", True, "HD95 mm (tumour-positive)"),
                "iou": boot(a, m0, "iou", True, "IoU (tumour-positive)"),
                "n_comp": boot(a, m0, "n_comp", False, "components/case (all)"),
            }
        # Swin against the single nnU-Net, connectivity only
        sw = Path(r"D:\data\swin") / split
        nn = Path(r"D:\data\nnuf0") / split
        if sw.exists() and nn.exists():
            print(f"  Swin f0 vs nnU-Net f0:")
            A, B = from_maps(sw), from_maps(nn)
            res[split]["Swin vs nnU-Net f0"] = {
                "n_comp": boot(A, B, "n_comp", False, "components/case (all)"),
                "comp_err": boot(A, B, "comp_err", False,
                                 "lesion count error (all)"),
            }
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
