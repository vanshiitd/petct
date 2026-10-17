#!/usr/bin/env python3
"""Phase IV step 1: failure diagnosis from ground truth (plan section 9).

Builds, for one split, a per-component and per-lesion record of what M0 got
wrong, using the failure classes fixed in `reports/phase4_analysis_plan.md`:

  spurious           predicted component with no ground-truth overlap, located
                     by majority TotalSegmentator label into physiological
                     uptake (brain, heart, bladder, kidneys), liver/spleen, other
  missed             ground-truth lesion touched by no predicted component
  under_seg          touched lesion, lesion Dice < 0.5 and matched predicted
                     volume < 0.6x the lesion
  over_seg           touched lesion, lesion Dice < 0.5 and matched predicted
                     volume > 1.67x the lesion
  fragmentation      one lesion touched by two or more components
  merging            one component touching two or more lesions
  boundary           touched lesion, lesion Dice in [0.5, 0.8), and mean
                     predictive entropy in the 2-voxel shell around the
                     component at least 1.5x its interior mean

The classes are not mutually exclusive, which is why the diagnosis model of
eq. 19 is multi-label.

Each class also gets an *oracle cost*: the case Dice that would be reached if
that class alone were corrected perfectly and everything else left as it is.
Fragmentation and merging move no voxels, so their Dice cost is zero by
construction and they are reported in lesion-count terms instead; the script
says so rather than printing a misleading zero.

Component ids match `build_candidates.py` and `exp4_refine.py` exactly -- same
map, same threshold, same 6-connectivity, same 5-voxel minimum -- so a row here
joins to a candidate row by (case, component_id).

    python scripts/phase4_diagnose.py --maps D:\\data\\cv\\oof \\
        --out D:\\data\\reports\\diag_oof.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
TOTALSEG = Path(r"D:\data\totalseg_ml_v2")
PHYSIO = [2, 3, 21, 51, 90]
LIVER, SPLEEN = 5, 1
MIN_VOX = 5

DICE_LOW, DICE_HI = 0.5, 0.8
UNDER_F, OVER_F = 0.6, 1.67
BAND_F = 1.5
CLASSES = ["spurious_physio", "spurious_liver_spleen", "spurious_other",
           "missed", "under_seg", "over_seg", "fragmentation", "merging",
           "boundary"]


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(case)


def crop_like(path: Path, bbox, shape) -> np.ndarray:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    a = a[z0:z1, y0:y1, x0:x1]
    if a.shape != shape:
        raise SystemExit(f"cropped {a.shape} != prepared {shape}")
    return a


def dice_of(inter: float, a: float, b: float) -> float:
    return (2.0 * inter / (a + b)) if (a + b) else float("nan")


def shell_ratio(mask_box: np.ndarray, ent_box: np.ndarray) -> float:
    """Mean entropy in the 2-voxel shell around a component over its interior."""
    outer = ndi.binary_dilation(mask_box, iterations=2)
    inner = ndi.binary_erosion(mask_box, iterations=2)
    shell = outer & ~inner
    if not shell.any() or not inner.any():
        return float("nan")
    return float(ent_box[shell].mean() / max(ent_box[inner].mean(), 1e-6))


def diagnose_case(case: str, maps: Path) -> dict:
    z = np.load(maps / f"{case}.npz")
    P = z["p_bar"].astype(np.float32)
    ent = z["u_ent"].astype(np.float32)
    stab = z["u_stab"].astype(np.float32)
    spacing = tuple(float(v) for v in z["spacing"])
    vml = float(np.prod(spacing)) / 1000.0
    shape = P.shape

    meta = json.loads((INV / f"{case}.json").read_text())
    bbox = meta["crop_bbox_zyx"]
    suv = crop_like(Path(meta["source_pet"]), bbox, shape).astype(np.float32)
    ts = TOTALSEG / f"{case}.nii.gz"
    organ = crop_like(ts, bbox, shape) if ts.exists() else None
    liver_suv = liver_sd = float("nan")
    if organ is not None and (organ == LIVER).any():
        lv = suv[organ == LIVER]
        liver_suv, liver_sd = float(lv.mean()), float(lv.std())

    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
    m0 = P > 0.5
    p_lab, p_n = ndi.label(m0)
    p_sizes = np.bincount(p_lab.ravel(), minlength=p_n + 1)
    comp_ids = [c for c in range(1, p_n + 1) if p_sizes[c] >= MIN_VOX]
    small_ids = [c for c in range(1, p_n + 1) if p_sizes[c] < MIN_VOX]
    g_lab, g_n = ndi.label(gt)
    g_sizes = np.bincount(g_lab.ravel(), minlength=g_n + 1)

    # overlap table: pred component -> {gt id: voxels}
    ov = {c: {} for c in range(1, p_n + 1)}
    both = m0 & gt
    if both.any():
        pairs, counts = np.unique(
            np.stack([p_lab[both], g_lab[both]]), axis=1, return_counts=True)
        for a, b, n in zip(pairs[0], pairs[1], counts):
            ov[int(a)][int(b)] = int(n)
    p_slices = ndi.find_objects(p_lab)

    comps, lesions = [], []
    for c in comp_ids:
        touched = ov[c]
        inter = sum(touched.values())
        sl = p_slices[c - 1]
        box = (p_lab[sl] == c)
        rec = {"case": case, "component_id": c, "n_vox": int(p_sizes[c]),
               "vol_ml": float(p_sizes[c] * vml), "inter": int(inter),
               "gt_ids": sorted(touched), "n_gt_touched": len(touched),
               "gt_overlap": {int(g): int(n) for g, n in touched.items()},
               "suv_max": float(suv[p_lab == c].max()),
               "shell_ratio": shell_ratio(box, ent[sl]),
               "ustab_mean": float(stab[p_lab == c].mean()),
               "ent_mean": float(ent[p_lab == c].mean())}
        if organ is not None:
            vals = organ[p_lab == c]
            nz = vals[vals > 0]
            rec["organ"] = int(np.bincount(nz).argmax()) if nz.size else 0
            rec["frac_physio"] = float(np.isin(vals, PHYSIO).mean())
            rec["frac_liver_spleen"] = float(np.isin(vals, [LIVER, SPLEEN]).mean())
        else:
            rec["organ"], rec["frac_physio"], rec["frac_liver_spleen"] = -1, np.nan, np.nan
        rec["spurious"] = inter == 0
        if inter == 0:
            if rec["frac_physio"] == rec["frac_physio"] and rec["frac_physio"] >= 0.5:
                rec["spurious_class"] = "spurious_physio"
            elif rec["frac_liver_spleen"] == rec["frac_liver_spleen"] \
                    and rec["frac_liver_spleen"] >= 0.5:
                rec["spurious_class"] = "spurious_liver_spleen"
            else:
                rec["spurious_class"] = "spurious_other"
        else:
            rec["spurious_class"] = None
        rec["merging"] = len(touched) >= 2
        comps.append(rec)

    # lesion side
    touch_of_gt = {g: [] for g in range(1, g_n + 1)}
    for c in comp_ids:
        for g in ov[c]:
            touch_of_gt[g].append(c)
    for g in range(1, g_n + 1):
        S = touch_of_gt[g]
        gvox = int(g_sizes[g])
        pvox = int(sum(p_sizes[c] for c in S))
        inter = int(sum(ov[c].get(g, 0) for c in S))
        d = dice_of(inter, pvox, gvox)
        sr = np.nan
        if S:
            c0 = max(S, key=lambda c: ov[c].get(g, 0))
            sl = p_slices[c0 - 1]
            sr = shell_ratio(p_lab[sl] == c0, ent[sl])
        lesions.append({
            "case": case, "gt_id": g, "n_vox": gvox, "vol_ml": float(gvox * vml),
            "n_comp": len(S), "comp_ids": S, "pred_vox": pvox, "inter": inter,
            "lesion_dice": d, "shell_ratio": float(sr),
            "suv_max": float(suv[g_lab == g].max()),
            "missed": len(S) == 0,
            "fragmentation": len(S) >= 2,
            "under_seg": bool(S and d < DICE_LOW and pvox < UNDER_F * gvox),
            "over_seg": bool(S and d < DICE_LOW and pvox > OVER_F * gvox),
            "boundary": bool(S and DICE_LOW <= d < DICE_HI
                             and sr == sr and sr >= BAND_F)})

    # oracle cost per class: fix that class alone, leave the rest
    keep_small = int(sum(p_sizes[c] for c in small_ids))
    inter_small = int(sum(sum(ov[c].values()) for c in small_ids))
    base_p = int(m0.sum())
    base_i = int(both.sum())
    gt_tot = int(gt.sum())
    base_dice = dice_of(base_i, base_p, gt_tot)

    def dice_after(dp: int, di: int) -> float:
        return dice_of(base_i + di, base_p + dp, gt_tot)

    oracle = {}
    sp = {k: [r for r in comps if r["spurious_class"] == k]
          for k in ("spurious_physio", "spurious_liver_spleen", "spurious_other")}
    for k, rs in sp.items():
        oracle[k] = dice_after(-sum(r["n_vox"] for r in rs), 0)
    miss = [l for l in lesions if l["missed"]]
    oracle["missed"] = dice_after(sum(l["n_vox"] for l in miss),
                                  sum(l["n_vox"] for l in miss))
    # Perfectly segmenting a flagged lesion means replacing every component that
    # touches it. A component can touch several lesions, so the removals and the
    # additions are taken over SETS -- summing per lesion would subtract a shared
    # component once per lesion and can drive the "oracle" Dice above 1.
    by_id = {c["component_id"]: c for c in comps}
    for k in ("under_seg", "over_seg", "boundary"):
        sel = [l for l in lesions if l[k]]
        rm = set()
        for l in sel:
            rm.update(l["comp_ids"])
        add = set()
        for c in rm:
            add.update(by_id[c]["gt_ids"])
        add.update(l["gt_id"] for l in sel)
        # voxels that components we are NOT removing already place inside the
        # lesions we are about to perfect: they are absorbed by the perfect
        # mask, so they must not be counted twice
        keep_in_add = sum(n for cid, cc in by_id.items() if cid not in rm
                          for g, n in cc["gt_overlap"].items() if g in add)
        keep_in_add += sum(ov[cid].get(g, 0) for cid in small_ids for g in add)
        gt_add = sum(int(g_sizes[g]) for g in add)
        dp = -sum(by_id[c]["n_vox"] for c in rm) - keep_in_add + gt_add
        di = -sum(by_id[c]["inter"] for c in rm) - keep_in_add + gt_add
        oracle[k] = dice_after(dp, di)
    oracle["fragmentation"] = base_dice       # no voxels move
    oracle["merging"] = base_dice

    return {"case": case, "vml": vml, "spacing": spacing, "gt_positive": bool(gt.any()),
            "gt_vox": gt_tot, "pred_vox": base_p, "inter": base_i,
            "base_dice": base_dice, "liver_suv": liver_suv, "liver_sd": liver_sd,
            "n_small": len(small_ids), "small_vox": keep_small,
            "small_inter": inter_small,
            "components": comps, "lesions": lesions, "oracle_dice": oracle}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.maps.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} cases from {args.maps}", flush=True)
    recs = []
    for i, f in enumerate(files, 1):
        recs.append(diagnose_case(f.stem, args.maps))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(recs, fh)

    pos = [r for r in recs if r["gt_positive"]]
    print(f"\n{len(recs)} cases ({len(pos)} tumour-positive) -> {args.out}")
    ncomp = sum(len(r["components"]) for r in recs)
    nles = sum(len(r["lesions"]) for r in recs)
    print(f"{ncomp} predicted components, {nles} ground-truth lesions")
    print(f"\n{'class':24s}{'cases':>8s}{'items':>8s}{'mean Dice':>11s}"
          f"{'oracle':>9s}{'cost':>8s}")
    base = float(np.mean([r["base_dice"] for r in pos]))
    for k in CLASSES:
        if k.startswith("spurious"):
            items = sum(sum(1 for c in r["components"]
                            if c["spurious_class"] == k) for r in recs)
            cs = [r for r in pos if any(c["spurious_class"] == k
                                        for c in r["components"])]
        elif k == "merging":            # component-level, not lesion-level
            items = sum(sum(1 for c in r["components"] if c["merging"])
                        for r in recs)
            cs = [r for r in pos if any(c["merging"] for c in r["components"])]
        else:
            items = sum(sum(1 for l in r["lesions"] if l[k]) for r in recs)
            cs = [r for r in pos if any(l[k] for l in r["lesions"])]
        if not cs:
            print(f"{k:24s}{0:8d}{items:8d}{'':>11s}{'':>9s}{'':>8s}")
            continue
        d0 = float(np.mean([r["base_dice"] for r in cs]))
        d1 = float(np.mean([r["oracle_dice"][k] for r in cs]))
        note = "  (topology only, no voxels move)" if k in ("fragmentation",
                                                            "merging") else ""
        print(f"{k:24s}{len(cs):8d}{items:8d}{d0:11.4f}{d1:9.4f}"
              f"{d1-d0:+8.4f}{note}")
    print(f"\nmean Dice over {len(pos)} tumour-positive cases: {base:.4f}")


if __name__ == "__main__":
    main()
