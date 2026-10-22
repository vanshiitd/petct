#!/usr/bin/env python3
"""Audit R4: the Experiment 1 metrics never reported for the later models.

IoU, ASSD, signed and median relative volume error, and connectivity are
computed with `evaluate_predictions.py`'s own functions - the same code that
produced every Experiment 1 number - applied to masks taken from the stored
probability maps at the 0.5 threshold, so no NIfTI has to be written.

Covers the three model outputs that exist as probability maps. The two agent
configurations are reported as not executed: the agents write per-case metrics
rather than masks, so their rows need a mask-saving re-run, which is noted in
the report rather than guessed at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from evaluate_predictions import (iou, assd_mm, hd95_mm, dice,
                                  component_counts)

MODELS = {
    "single 505 (fold 0, logit 8-flip)": r"D:\data\uncertainty_m0L",
    "M0 (5-fold ensemble)": r"D:\data\cv",
    "Swin f0": r"D:\data\swin",
}


def run(maps: Path, limit=None) -> list[dict]:
    cases = sorted(f.stem for f in maps.glob("*.npz"))
    if limit:
        cases = cases[:limit]
    rows = []
    for i, c in enumerate(cases, 1):
        ctx = CaseContext(c, maps)
        p, g, sp = ctx.mask(), ctx.gt, ctx.spacing
        vox = float(np.prod(sp)) / 1000.0
        npred, ngt = component_counts(p, g)
        pv, gv = float(p.sum()) * vox, float(g.sum()) * vox
        rows.append({
            "case": c, "dice": dice(p, g) if g.any() else float("nan"),
            "iou": iou(p, g) if g.any() else float("nan"),
            "assd": (assd_mm(p, g, sp) if p.any() and g.any() else float("nan")),
            "hd95": (hd95_mm(p, g, sp) if p.any() and g.any() else float("nan")),
            "vol_err_ml": pv - gv,
            "rel_vol_err": ((pv - gv) / gv) if gv > 0 else float("nan"),
            "n_pred_comp": npred, "n_gt_comp": ngt,
            "comp_err": npred - ngt, "gt_positive": int(g.any()),
        })
        if i % 50 == 0:
            print(f"    {i}/{len(cases)}", flush=True)
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    pos = [r for r in rows if r["gt_positive"]]
    f = lambda k, src=pos: float(np.nanmean([r[k] for r in src]))  # noqa: E731
    out = {
        "label": label, "n": len(rows), "n_pos": len(pos),
        "dice": f("dice"), "iou": f("iou"), "assd_mm": f("assd"),
        "hd95_mm": f("hd95"),
        "vol_err_ml_signed": float(np.nanmean([r["vol_err_ml"] for r in rows])),
        "rel_vol_err_median": float(np.nanmedian([r["rel_vol_err"] for r in pos])),
        "n_pred_comp_mean": float(np.mean([r["n_pred_comp"] for r in rows])),
        "lesion_count_err_mean": float(np.mean([r["comp_err"] for r in rows])),
    }
    print(f"  {label:34s} IoU {out['iou']:.4f}  ASSD {out['assd_mm']:7.2f}  "
          f"volerr {out['vol_err_ml_signed']:+8.2f} mL  "
          f"relvol(med) {100*out['rel_vol_err_median']:+7.1f}%  "
          f"comps {out['n_pred_comp_mean']:6.2f}  "
          f"count err {out['lesion_count_err_mean']:+6.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_exp1_metrics.json"))
    args = ap.parse_args()
    res = {}
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        res[split] = {}
        for label, root in MODELS.items():
            maps = Path(root) / split
            if not maps.exists():
                print(f"  {label:34s} maps not found at {maps}")
                continue
            res[split][label] = summarise(run(maps, args.limit), label)
    res["not_executed"] = (
        "Phase V agent and final agent: the agents write per-case metrics, not "
        "masks, so these rows need a mask-saving re-run (about 40 minutes for "
        "val+test per configuration). Reported as not executed rather than "
        "approximated.")
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
