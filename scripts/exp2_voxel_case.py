#!/usr/bin/env python3
"""Experiment 2, voxel and case level: does uncertainty predict error?

Hypothesis H1 from the work plan is that uncertainty is positively correlated
with actual segmentation error. This tests it two ways per case, in one pass
over the maps.

**Voxel level.** An error voxel is one where the ensemble segmentation
(P_bar > 0.5) disagrees with the truth -- a false positive or a false negative,
i.e. pred XOR gt. AUROC of each uncertainty map for separating error from
non-error voxels is accumulated as a histogram rather than by keeping voxels:
200 test cases at ~13 million voxels each cannot be held in memory, and
subsampling would put a confidence interval on a quantity that can be computed
exactly. Binning the score into 2048 fixed bins and counting errors per bin
gives the AUROC to within one bin width, with memory that does not depend on
the number of cases.

Two candidate regions, because the choice changes the answer and quoting only
one would overstate it:

AUROC is estimated from a stratified voxel sample, not from the histograms. A
2048-bin histogram looked attractive -- exact, constant memory -- but these
distributions saturate: 99.2% of body voxels sit in one bin because P_bar is
essentially 0 almost everywhere, and entropy and confidence compress that mass
differently (9% against 21% of the error voxels land in the saturated bin). Two
scores that are provably the same ranking then score 0.95 and 0.89. Binning is
therefore unusable here, and sampling is used instead: every error voxel is
kept, non-error voxels are kept with a fixed per-class probability, and a
two-sample Mann-Whitney estimate on that sample is unbiased for the population
AUROC regardless of how the score is parameterised. The histograms are kept for
calibration only, where bins are coarse and the problem does not arise.

Candidate regions:

  * **body** -- everything inside the body (original CT > -500 HU, cropped into
    the prepared grid). This is the honest denominator for "could the model
    have gone wrong here", but it is dominated by easy background, which
    inflates AUROC.
  * **near** -- within 10 mm of the predicted or true tumour. This is the hard
    region where a reader would actually be looking, and the AUROC there is the
    one worth believing.

**Calibration.** Reliability bins and ECE for P_bar inside each region, plus the
counts needed to fit a single temperature on val and re-score.

**Case level.** Per-case uncertainty summaries (mean and sum of each U inside
the prediction, and inside the near region) against 1 - Dice, FP volume and
HD95, written per case so the correlations and tertile table are computed later
from val cut-offs.

Nothing here chooses a threshold: this script only produces counts and per-case
numbers. `exp2_report.py` fits on val and applies to test.

    python scripts/exp2_voxel_case.py --maps D:\\data\\uncertainty\\val \\
        --split val --out-prefix D:\\data\\reports\\exp2_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
NBINS = 2048
# Sampling rates per class. Error voxels are always kept; non-error voxels are
# thinned so the pooled sample fits in memory. The rate depends only on the
# class and the region, never on the case, which is what keeps the pooled
# sample uniform within class and the AUROC estimate unbiased.
KEEP_NONERR = {"body": 0.005, "near": 1.0}
# fixed ranges so histograms from different cases and different runs are
# commensurable; U_ent is bounded by ln 2, the two variances by 0.25
RANGES = {"u_ent": (0.0, float(np.log(2)) + 1e-6),
          "u_epi": (0.0, 0.25 + 1e-6),
          "u_stab": (0.0, 0.25 + 1e-6),
          "conf": (0.5, 1.0 + 1e-6),
          "p_bar": (0.0, 1.0 + 1e-6)}
U_KEYS = ("u_ent", "u_epi", "u_stab", "conf")


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(f"no prepared label for {case}")


def body_mask(case: str, shape: tuple) -> np.ndarray:
    """Body from the ORIGINAL CT in HU, cropped into the prepared grid.

    The prepared CT channel is per-scan z-scored, so -500 HU has no meaning in
    it. The crop recorded in the inverse metadata has zoom 1.0, so taking the
    original CT and slicing it is exact, not an approximation.
    """
    meta = json.loads((INV / f"{case}.json").read_text())
    ct_path = Path(meta["source_pet"]).with_name("CT_resample.nii.gz")
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
    (z0, z1), (y0, y1), (x0, x1) = meta["crop_bbox_zyx"]
    ct = ct[z0:z1, y0:y1, x0:x1]
    if ct.shape != shape:
        raise SystemExit(f"{case}: cropped CT {ct.shape} != maps {shape}")
    return ct > -500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--near-mm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.maps.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} cases from {args.maps}\n", flush=True)

    rng = np.random.default_rng(2026)
    regions = ("body", "near")
    samples = {r: {k: [] for k in list(RANGES) + ["err"]} for r in regions}
    # hist[region][key] = (counts over bins for error, counts for non-error)
    hist = {r: {k: np.zeros((2, NBINS), dtype=np.int64) for k in RANGES} for r in regions}
    rows = []

    for i, f in enumerate(files, 1):
        case = f.stem
        z = np.load(f)
        p_bar = z["p_bar"].astype(np.float32)
        u = {k: z[k].astype(np.float32) for k in ("u_ent", "u_epi", "u_stab")}
        u["conf"] = np.maximum(p_bar, 1.0 - p_bar)
        u["p_bar"] = p_bar
        spacing = tuple(float(v) for v in z["spacing"])   # (z, y, x) in mm

        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
        pred = p_bar > 0.5
        err = pred ^ gt

        body = body_mask(case, p_bar.shape)
        union = pred | gt
        if union.any():
            near = ndi.distance_transform_edt(~union, sampling=spacing) <= args.near_mm
        else:
            near = np.zeros_like(body)
        near &= body            # "near the tumour" still means inside the body

        for rname, region in (("body", body), ("near", near)):
            if not region.any():
                continue
            e = err[region]
            for k, arr in u.items():
                lo, hi = RANGES[k]
                b = np.clip(((arr[region] - lo) / (hi - lo) * NBINS).astype(np.int32),
                            0, NBINS - 1)
                hist[rname][k][0] += np.bincount(b[~e], minlength=NBINS)
                hist[rname][k][1] += np.bincount(b[e], minlength=NBINS)

            # stratified sample for AUROC: all errors, a fixed fraction of the rest
            q = KEEP_NONERR[rname]
            if q >= 1.0:
                take = np.ones(e.shape, dtype=bool)      # keep the whole region
            else:
                take = e | ((~e) & (rng.random(e.shape) < q))
            idx = np.flatnonzero(take)
            for k, arr in u.items():
                samples[rname][k].append(arr[region][idx].astype(np.float16))
            samples[rname]["err"].append(e[idx].astype(np.uint8))

        voxel_ml = float(np.prod(spacing)) / 1000.0
        row = {"case": case, "gt_positive": int(gt.any()), "pred_positive": int(pred.any()),
               "gt_ml": float(gt.sum() * voxel_ml), "pred_ml": float(pred.sum() * voxel_ml),
               "err_ml": float(err.sum() * voxel_ml),
               "body_voxels": int(body.sum()), "near_voxels": int(near.sum()),
               "err_voxels_body": int(err[body].sum()),
               "err_voxels_near": int(err[near].sum()) if near.any() else 0}
        for k in U_KEYS:
            row[f"{k}_mean_pred"] = float(u[k][pred].mean()) if pred.any() else float("nan")
            row[f"{k}_max_pred"] = float(u[k][pred].max()) if pred.any() else float("nan")
            row[f"{k}_sum_pred"] = float(u[k][pred].sum() * voxel_ml) if pred.any() else 0.0
            row[f"{k}_mean_near"] = float(u[k][near].mean()) if near.any() else float("nan")
            row[f"{k}_sum_near"] = float(u[k][near].sum() * voxel_ml) if near.any() else 0.0
            row[f"{k}_mean_body"] = float(u[k][body].mean())
            row[f"{k}_sum_body"] = float(u[k][body].sum() * voxel_ml)
        rows.append(row)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    out_h = args.out_prefix.with_name(args.out_prefix.name + "_hist.npz")
    np.savez_compressed(out_h, nbins=NBINS,
                        ranges=json.dumps(RANGES),
                        **{f"{r}__{k}": hist[r][k] for r in regions for k in RANGES})
    out_s = args.out_prefix.with_name(args.out_prefix.name + "_samples.npz")
    packed = {}
    for r in regions:
        for k, chunks in samples[r].items():
            packed[f"{r}__{k}"] = (np.concatenate(chunks) if chunks
                                   else np.zeros(0, dtype=np.float16))
    np.savez_compressed(out_s, keep_nonerr=json.dumps(KEEP_NONERR), **packed)
    for r in regions:
        n = packed[f"{r}__err"].size
        ne = int(packed[f"{r}__err"].sum()) if n else 0
        print(f"  {r}: sampled {n:,} voxels ({ne:,} error, {n - ne:,} non-error)")

    out_c = args.out_prefix.with_name(args.out_prefix.name + "_cases.json")
    out_c.write_text(json.dumps(rows, indent=1))
    print(f"\nhistograms -> {out_h}\nper-case    -> {out_c}")


if __name__ == "__main__":
    main()
