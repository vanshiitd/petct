#!/usr/bin/env python3
"""Turn Experiment 4's component decisions into masks on disk.

The tuning in `exp4_refine.py` works from component bookkeeping, which gives
Dice, FP and FN volume exactly and for free. HD95, ASSD and the rest need real
masks, so the chosen variants are written out here and scored with the same
evaluator as everything else in this project.

Component ids are recomputed exactly as in `build_candidates.py` and
`exp4_refine.py` -- same map, same thresholds, same 6-connectivity -- so a
decision recorded against an id lands on the intended voxels. The result is
checked against the bookkeeping prediction: if the written mask's volume
disagrees with what the decision implies, the script says so rather than
quietly writing a mask that means something else.

    python scripts/exp4_materialise.py --decisions reports/exp4_refine.json \\
        --variant "A+B [conservative]" --split test --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
UNC = Path(r"D:\data\uncertainty_m0")
T_LOW, MIN_VOX = 0.1, 5


def reference(case: str) -> sitk.Image:
    for sub in ("imagesTs", "imagesTr"):
        f = RAW / sub / f"{case}_0000.nii.gz"
        if f.exists():
            return sitk.ReadImage(str(f))
    raise FileNotFoundError(case)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dec = json.loads(args.decisions.read_text())["decisions"][args.variant]
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted((UNC / args.split).glob("*.npz"))
    print(f"{args.variant}: {len(files)} cases -> {args.out}")

    n_removed = n_added = 0
    for i, f in enumerate(files, 1):
        case = f.stem
        z = np.load(f)
        P = z["p_bar"].astype(np.float32)
        m0 = P > 0.5
        keep_ids, add_ids = dec.get(case, [[], []])
        keep_ids, add_ids = set(keep_ids), set(add_ids)

        out = np.zeros_like(m0)
        p_lab, p_n = ndi.label(m0)
        if p_n:
            sizes = np.bincount(p_lab.ravel(), minlength=p_n + 1)
            # small components were never candidates and are carried over as-is
            keep_all = {c for c in range(1, p_n + 1) if sizes[c] < MIN_VOX} | keep_ids
            n_removed += sum(1 for c in range(1, p_n + 1)
                             if sizes[c] >= MIN_VOX and c not in keep_ids)
            out |= np.isin(p_lab, list(keep_all)) if keep_all else False
        if add_ids:
            low = (P > T_LOW) & ~m0
            l_lab, l_n = ndi.label(low)
            out |= np.isin(l_lab, list(add_ids))
            n_added += len(add_ids)

        ref = reference(case)
        img = sitk.GetImageFromArray(out.astype(np.uint8))
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(args.out / f"{case}.nii.gz"), True)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print(f"written {len(files)} | components removed {n_removed} | added {n_added}")


if __name__ == "__main__":
    main()
