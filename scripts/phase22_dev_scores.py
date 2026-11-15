#!/usr/bin/env python3
"""Phase XXII development: label-free per-patient disagreement features between
the deployed mask and the fine-tuned segmenter, plus the outcome columns needed
to judge them.

  --split A   development (default). Uses p22_masks_A.
  --split B / C  only after the mechanism is frozen (see the plan).

Features use no ground truth: they come from the deployed mask, the fine-tuned
probability map and the zero-shot probability map. Outcome columns (hits, Dice,
harm) are computed separately and written next to them.

Writes phase22_scores_{split}.json.
"""
from __future__ import annotations

import argparse
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
MASKS = {"A": "p22_masks_A", "B": "p21_masks_B", "C": "p21_masks_C"}
VBINS = [0.5, 2, 5, 10, 20, 50, 1e9]          # mL upper edges


def dice(m, gt):
    s = int(m.sum()) + int(gt.sum())
    return 2 * int((m & gt).sum()) / s if s else float("nan")


def features(p, p0, ref, vml):
    """Per-patient, label-free. Components of the deployed mask that the
    fine-tuned mask does not touch ('orphans'), by volume bin."""
    ft = p > 0.5
    lab, n = ndi.label(ref)
    f = {"ref_ml": float(ref.sum() * vml), "ft_ml": float(ft.sum() * vml),
         "n_ref": int(n), "n_ft": int(ndi.label(ft)[1])}
    if n == 0:
        return {**f, "orph": [0] * len(VBINS), "orph_hi": [0] * len(VBINS),
                "orph_ml": 0.0, "kept_frac": 1.0}
    idx = np.arange(1, n + 1)
    size = ndi.sum(np.ones_like(lab), lab, idx) * vml
    touch = ndi.maximum(ft, lab, idx) > 0
    pmax = ndi.maximum(p, lab, idx)
    p0mean = ndi.mean(p0, lab, idx)
    orph = ~touch
    b = np.searchsorted(VBINS, size)
    f["orph"] = [int((orph & (b == i)).sum()) for i in range(len(VBINS))]
    # orphans the zero-shot model was confident about
    f["orph_hi"] = [int((orph & (b == i) & (p0mean >= 0.8)).sum()) for i in range(len(VBINS))]
    f["orph_ml"] = float(size[orph].sum())
    f["orph_pmax_lt01"] = int((orph & (pmax < 0.01)).sum())
    f["kept_frac"] = float(touch.mean())
    # deployed volume inside components that FT touches but mostly rejects
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A")
    s = ap.parse_args().split
    cases = json.loads((REPO / "reports" / "phase19_split.json").read_text())[s]
    rows = []
    for i, c in enumerate(cases, 1):
        z = np.load(FT / f"{c}.npz")
        p = z["p_bar"].astype(np.float32)
        vml = float(np.prod(z["spacing"])) / 1000.0
        p0 = np.load(M0 / f"{c}.npz")["p_bar"].astype(np.float32)
        ref = np.load(R / MASKS[s] / f"{c}.npz")["mask"].astype(bool)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
        ft = p > 0.5
        g_lab, g_n = ndi.label(gt)
        hr = set(np.unique(g_lab[ref]).tolist()) - {0}
        hf = set(np.unique(g_lab[ft]).tolist()) - {0}
        dr, df = dice(ref, gt), dice(ft, gt)
        rows.append({"case": c, **features(p, p0, ref, vml),
                     "n_gt": g_n, "hit_ref": len(hr), "hit_ft": len(hf),
                     "n_lost": len(hr - hf), "n_gained": len(hf - hr),
                     "dice_ref": dr, "dice_ft": df,
                     "harmed": bool(len(hf) < len(hr) or (g_n > 0 and df - dr < -0.01))})
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)
    (R / f"phase22_scores_{s}.json").write_text(json.dumps(rows, indent=1))
    print(f"written phase22_scores_{s}.json; harmed {sum(r['harmed'] for r in rows)}")


if __name__ == "__main__":
    main()
