#!/usr/bin/env python3
"""Phase XXII development (split A by default): the fine-tuned map at lower
thresholds, FT_t = {p_FT > t}, against the deployed reference mask.
Harm is the unchanged definition. Writes phase22_thresh_{split}.json."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import label_path

R = Path(r"D:\data\reports")
REPO = Path(__file__).resolve().parent.parent
FT = Path(r"C:\psma_maps\ft")
MASKS = {"A": "p22_masks_A", "B": "p21_masks_B", "C": "p21_masks_C"}
TS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]


def sc(m, gt, g_lab):
    s = int(m.sum()) + int(gt.sum())
    return (2 * int((m & gt).sum()) / s if s else float("nan"),
            len(set(np.unique(g_lab[m]).tolist()) - {0}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="A")
    s = ap.parse_args().split
    cases = json.loads((REPO / "reports" / "phase19_split.json").read_text())[s]
    out = {t: [] for t in TS}
    for c in cases:
        z = np.load(FT / f"{c}.npz"); p = z["p_bar"].astype(np.float32)
        vml = float(np.prod(z["spacing"])) / 1000.0
        ref = np.load(R / MASKS[s] / f"{c}.npz")["mask"].astype(bool)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
        g_lab, g_n = ndi.label(gt)
        dr, hr = sc(ref, gt, g_lab)
        for t in TS:
            m = p > t
            d, h = sc(m, gt, g_lab)
            out[t].append({"case": c, "n_gt": g_n, "dice": d, "hit": h, "dice_ref": dr,
                           "hit_ref": hr, "fp_ml": float((m & ~gt).sum() * vml),
                           "harmed": bool(h < hr or (g_n > 0 and d - dr < -0.01))})
    print(f"[{s}] n = {len(cases)}")
    for t in TS:
        r = out[t]; pos = [x for x in r if x["n_gt"] > 0]
        print(f"  t {t:4.2f} harmed {sum(x['harmed'] for x in r):2d} (lost-hit "
              f"{sum(x['hit'] < x['hit_ref'] for x in r)}) Dice {np.mean([x['dice'] for x in pos]):.4f} "
              f"hits {sum(x['hit'] for x in r)} FPvox mL {np.mean([x['fp_ml'] for x in r]):.1f}")
    (R / f"phase22_thresh_{s}.json").write_text(json.dumps({str(t): v for t, v in out.items()}))


if __name__ == "__main__":
    main()
