#!/usr/bin/env python3
"""Phase XXII development, split A only: what do the lesions look like that the
fine-tuned segmenter loses relative to the deployed agent?

A is the adaptation split. Its fine-tuned predictions are out-of-fold. Nothing
here reads B, C or the follow-up studies.

For every A patient: Dice and lesion hits of the reference (hand-built agent on
zero-shot maps) and of FT (p > 0.5), the harm indicator, and one row per ground-
truth lesion that the reference hits and FT misses.

Writes phase22_dev_diag_A.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import label_path

R = Path(r"D:\data\reports")
REPO = Path(__file__).resolve().parent.parent
FT, M0 = Path(r"C:\psma_maps\ft"), Path(r"D:\data\cv\psma")


def load(c, split="A"):
    z = np.load(FT / f"{c}.npz")
    p = z["p_bar"].astype(np.float32)
    vml = float(np.prod(z["spacing"])) / 1000.0
    p0 = np.load(M0 / f"{c}.npz")["p_bar"].astype(np.float32)
    ref = np.load(R / f"p22_masks_{split}" / f"{c}.npz")["mask"].astype(bool)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
    return p, p0, ref, gt, vml


def dice(m, gt):
    s = int(m.sum()) + int(gt.sum())
    return 2 * int((m & gt).sum()) / s if s else float("nan")


def main() -> None:
    cases = json.loads((REPO / "reports" / "phase19_split.json").read_text())["A"]
    pats, lost = [], []
    for i, c in enumerate(cases, 1):
        p, p0, ref, gt, vml = load(c)
        ft = p > 0.5
        g_lab, g_n = ndi.label(gt)
        r_lab, _ = ndi.label(ref)
        r_sizes = np.bincount(r_lab.ravel())
        hit_r = set(np.unique(g_lab[ref]).tolist()) - {0}
        hit_f = set(np.unique(g_lab[ft]).tolist()) - {0}
        dr, df = dice(ref, gt), dice(ft, gt)
        harmed = len(hit_f) < len(hit_r) or (g_n > 0 and df - dr < -0.01)
        pats.append({"case": c, "n_gt": g_n, "hit_ref": len(hit_r), "hit_ft": len(hit_f),
                     "dice_ref": dr, "dice_ft": df, "harmed": bool(harmed),
                     "ref_ml": float(ref.sum() * vml), "ft_ml": float(ft.sum() * vml),
                     "gt_ml": float(gt.sum() * vml), "n_lost": len(hit_r - hit_f),
                     "n_gained": len(hit_f - hit_r)})
        for l in sorted(hit_r - hit_f):
            lm = g_lab == l
            comps = [k for k in np.unique(r_lab[lm]) if k]
            big = max(comps, key=lambda k: r_sizes[k])
            cm = r_lab == big
            lost.append({"case": c, "lesion_ml": float(lm.sum() * vml),
                         "ref_cover": float((lm & ref).sum() / lm.sum()),
                         "ref_comp_ml": float(r_sizes[big] * vml),
                         "ref_comp_gt_frac": float((cm & gt).sum() / cm.sum()),
                         "pft_max_lesion": float(p[lm].max()),
                         "pft_max_comp": float(p[cm].max()),
                         "p0_max_lesion": float(p0[lm].max()),
                         "p0_mean_lesion": float(p0[lm].mean())})
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)
    h = [x for x in pats if x["harmed"]]
    print(f"A: {len(pats)} patients, harmed {len(h)} (lost a hit: "
          f"{sum(x['n_lost'] > 0 and x['hit_ft'] < x['hit_ref'] for x in pats)}), "
          f"lost lesions {len(lost)}")
    pos = [x for x in pats if x["n_gt"] > 0]
    print(f"mean Dice (tumour-positive): ref {np.mean([x['dice_ref'] for x in pos]):.4f} "
          f"FT {np.mean([x['dice_ft'] for x in pos]):.4f}")
    for x in h:
        print(f"  {x['case'][:21]} hits {x['hit_ref']}->{x['hit_ft']} of {x['n_gt']} "
              f"(lost {x['n_lost']}, gained {x['n_gained']}) dice {x['dice_ref']:.3f}->"
              f"{x['dice_ft']:.3f} ref {x['ref_ml']:.0f} mL ft {x['ft_ml']:.0f} mL")
    if lost:
        for k in ("lesion_ml", "ref_cover", "ref_comp_ml", "ref_comp_gt_frac",
                  "pft_max_lesion", "p0_max_lesion"):
            v = np.array([r[k] for r in lost])
            print(f"  lost lesions {k:18s} median {np.median(v):9.3f} "
                  f"q10 {np.percentile(v, 10):9.3f} q90 {np.percentile(v, 90):9.3f}")
    (R / "phase22_dev_diag_A.json").write_text(
        json.dumps({"patients": pats, "lost": lost}, indent=1))


if __name__ == "__main__":
    main()
