#!/usr/bin/env python3
"""Audit R4, agent rows: Experiment 1 metrics on the agents' own masks.

`audit_exp1_metrics.py` covers the three base models, whose output exists as
probability maps. The agents write per-case metrics rather than masks, so this
reads the masks saved by `phase5_agent.py --save-masks` and scores them with the
same `evaluate_predictions.py` functions, so an agent row and a base-model row
in the R4 table come from one metric implementation.

Ground truth comes through `CaseContext`, which is also where the base-model
rows got theirs.
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

MAPS = {"val": Path(r"D:\data\cv\val"), "test": Path(r"D:\data\cv\test")}
RUNS = {
    "Phase V agent": "pv",
    "final agent": "fa",
}


def score_dir(mask_dir: Path, maps: Path) -> list[dict]:
    rows = []
    files = sorted(mask_dir.glob("*.npz"))
    for i, f in enumerate(files, 1):
        case = f.stem
        z = np.load(f)
        p = z["mask"].astype(bool)
        ctx = CaseContext(case, maps)
        g, sp = ctx.gt, ctx.spacing
        if p.shape != g.shape:
            raise SystemExit(f"{case}: mask {p.shape} vs gt {g.shape}")
        vox = float(np.prod(sp)) / 1000.0
        npred, ngt = component_counts(p, g)
        pv, gv = float(p.sum()) * vox, float(g.sum()) * vox
        rows.append({
            "case": case,
            "dice": dice(p, g) if g.any() else float("nan"),
            "iou": iou(p, g) if g.any() else float("nan"),
            "assd": (assd_mm(p, g, sp) if p.any() and g.any() else float("nan")),
            "hd95": (hd95_mm(p, g, sp) if p.any() and g.any() else float("nan")),
            "vol_err_ml": pv - gv,
            "rel_vol_err": ((pv - gv) / gv) if gv > 0 else float("nan"),
            "n_pred_comp": npred, "n_gt_comp": ngt,
            "comp_err": npred - ngt, "gt_positive": int(g.any()),
        })
        if i % 50 == 0:
            print(f"    {i}/{len(files)}", flush=True)
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    pos = [r for r in rows if r["gt_positive"]]
    f = lambda k: float(np.nanmean([r[k] for r in pos]))     # noqa: E731
    out = {"label": label, "n": len(rows), "n_pos": len(pos),
           "dice": f("dice"), "iou": f("iou"), "assd_mm": f("assd"),
           "hd95_mm": f("hd95"),
           "vol_err_ml_signed": float(np.nanmean([r["vol_err_ml"] for r in rows])),
           "rel_vol_err_median": float(np.nanmedian([r["rel_vol_err"] for r in pos])),
           "n_pred_comp_mean": float(np.mean([r["n_pred_comp"] for r in rows])),
           "lesion_count_err_mean": float(np.mean([r["comp_err"] for r in rows]))}
    print(f"  {label:34s} Dice {out['dice']:.4f}  IoU {out['iou']:.4f}  "
          f"ASSD {out['assd_mm']:7.2f}  HD95 {out['hd95_mm']:7.2f}  "
          f"volerr {out['vol_err_ml_signed']:+8.2f}  "
          f"relmed {100*out['rel_vol_err_median']:+7.1f}%  "
          f"comps {out['n_pred_comp_mean']:6.2f}  "
          f"cnterr {out['lesion_count_err_mean']:+6.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\data\masks"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\audit_mask_metrics.json"))
    args = ap.parse_args()
    res = {}
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        res[split] = {}
        for label, pre in RUNS.items():
            d = args.root / f"{pre}_{split}"
            if not d.exists() or not any(d.glob("*.npz")):
                print(f"  {label:34s} masks not found at {d}")
                continue
            res[split][label] = summarise(score_dir(d, MAPS[split]), label)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
