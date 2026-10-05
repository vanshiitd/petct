#!/usr/bin/env python3
"""Tune the post-processing thresholds on validation predictions.

Both rules act by deleting whole connected components, so every setting in the
grid keeps or drops components from the *same* decomposition. Each case is
therefore read once: its components are labelled, and for each component the
script records volume, the fraction inside an allowed organ, and the overlap
with the ground truth. Every (min volume, organ overlap) pair is then scored
from those three numbers, with no second pass over the images and no
intermediate files -- which is what makes a grid this size affordable.

Scoring is exact rather than approximate, because for a kept set K

    Dice = 2 * sum_K |component & gt| / (sum_K |component| + |gt|)

False positives and negatives follow `evaluate_predictions.py`'s definition, which
is at component level, not voxel level: false-positive volume is the volume of
predicted components touching *no* part of the ground truth, and false-negative
volume is the volume of ground-truth components touched by no kept prediction.
Those are the numbers every other report here quotes, so the sweep has to use the
same ones -- a voxel-level count of the same predictions gives FN 41 mL where the
component-level count gives 4 mL, and mixing the two would make the before/after
comparison meaningless.

Tuned on validation only. The chosen setting is then applied once to test.

    python scripts/sweep_postprocess.py --pred <val preds> --gt <val gt> \\
        --totalseg D:\\data\\totalseg_ml_v2 --organs kidney_right,kidney_left,urinary_bladder,brain \\
        --out reports/sweep_505_val.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

DEFAULT_LABEL_IDS = {
    "spleen": 1, "kidney_right": 2, "kidney_left": 3, "liver": 5,
    "urinary_bladder": 21, "heart": 51, "brain": 90,
}


def _geom(img) -> tuple:
    return (img.GetSize(),
            tuple(round(v, 6) for v in img.GetSpacing()),
            tuple(round(v, 3) for v in img.GetOrigin()),
            tuple(round(v, 6) for v in img.GetDirection()))


def summarise_case(job) -> dict:
    """Per-component volume, organ fraction and ground-truth overlap."""
    case, pred_path, gt_path, ts_path, organ_ids = job
    pred_img = sitk.ReadImage(str(pred_path))
    gt_img = sitk.ReadImage(str(gt_path))
    if _geom(pred_img) != _geom(gt_img):
        return {"case": case, "error": "prediction and ground truth geometry differ"}

    arr = sitk.GetArrayFromImage(pred_img) > 0
    gt = sitk.GetArrayFromImage(gt_img) > 0
    sx, sy, sz = pred_img.GetSpacing()
    voxel_ml = sx * sy * sz / 1000.0

    lab, n = ndi.label(arr)
    sizes = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    inter = np.bincount(lab[gt].ravel(), minlength=n + 1).astype(np.int64) if gt.any() \
        else np.zeros(n + 1, dtype=np.int64)

    # ground-truth components, and which predicted component touches each, so a
    # false negative can be scored against whatever survives the thresholds
    gt_lab, gt_n = ndi.label(gt)
    gt_sizes = np.bincount(gt_lab.ravel(), minlength=gt_n + 1).astype(np.int64)
    both = gt & arr
    pairs = sorted({(int(g), int(p)) for g, p in
                    zip(gt_lab[both].ravel(), lab[both].ravel())})

    organ_frac = np.zeros(n + 1)
    if organ_ids and n:
        ts_file = Path(ts_path)
        if not ts_file.exists():
            return {"case": case, "error": "no organ mask"}
        ts_img = sitk.ReadImage(str(ts_file))
        if _geom(ts_img) != _geom(pred_img):
            return {"case": case, "error": "organ mask geometry differs from prediction"}
        organ = np.isin(sitk.GetArrayFromImage(ts_img), list(organ_ids))
        in_organ = np.bincount(lab[organ].ravel(), minlength=n + 1).astype(np.int64)
        organ_frac = in_organ / np.maximum(sizes, 1)

    return {"case": case, "voxel_ml": voxel_ml, "gt_voxels": int(gt.sum()),
            "sizes": sizes[1:].tolist(), "inter": inter[1:].tolist(),
            "organ_frac": organ_frac[1:].tolist(),
            "gt_sizes": gt_sizes[1:].tolist(), "pairs": pairs}


def score(cases: list[dict], min_ml: float, overlap: float) -> dict:
    dices, fp_ml, fn_ml, neg_with_fp, empty_pos = [], [], [], 0, 0
    for c in cases:
        sizes = np.asarray(c["sizes"], dtype=np.int64)
        inter = np.asarray(c["inter"], dtype=np.int64)
        ofrac = np.asarray(c["organ_frac"])
        v = c["voxel_ml"]
        keep = np.ones(len(sizes), dtype=bool)
        if min_ml > 0:
            keep &= (sizes * v) >= min_ml
        if overlap <= 1.0:
            keep &= ofrac <= overlap
        pred_vox = int(sizes[keep].sum())
        tp = int(inter[keep].sum())
        gt_vox = c["gt_voxels"]

        # component-level, matching evaluate_predictions.py
        fp_ml.append(float(sizes[keep & (inter == 0)].sum()) * v)
        gt_sizes = np.asarray(c["gt_sizes"], dtype=np.int64)
        matched = {g for g, p in c["pairs"] if keep[p - 1]}
        unmatched = [gs for i, gs in enumerate(gt_sizes, start=1) if i not in matched]
        fn_ml.append(float(sum(unmatched)) * v)
        if gt_vox > 0:
            denom = pred_vox + gt_vox
            dices.append(2 * tp / denom if denom else 1.0)
            empty_pos += int(pred_vox == 0)
        elif pred_vox > 0:
            neg_with_fp += 1
    return {"dice": float(np.mean(dices)), "dice_median": float(np.median(dices)),
            "fp_ml": float(np.mean(fp_ml)), "fn_ml": float(np.mean(fn_ml)),
            "neg_with_fp": neg_with_fp, "empty_positive": empty_pos}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--totalseg", type=Path, default=None)
    p.add_argument("--organs", default="")
    p.add_argument("--label-ids", type=Path, default=None)
    p.add_argument("--min-volumes", default="0,0.1,0.2,0.5,1.0,2.0,3.0,5.0")
    p.add_argument("--overlaps", default="1.01,0.75,0.5,0.25,0.1")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ids = DEFAULT_LABEL_IDS
    if args.label_ids and args.label_ids.exists():
        ids = {k: int(v) for k, v in json.loads(args.label_ids.read_text()).items()}
    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    unknown = [o for o in organs if o not in ids]
    if unknown:
        raise SystemExit(f"unknown organ(s): {unknown}; have {sorted(ids)}")
    organ_ids = [ids[o] for o in organs]

    preds = sorted(args.pred.glob("*.nii.gz"))
    jobs = [(f.name[:-7], str(f), str(args.gt / f.name),
             str((args.totalseg or Path(".")) / f.name), organ_ids) for f in preds]
    print(f"{len(jobs)} cases | organs {organs or 'none'}\n")

    cases = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(summarise_case, jobs, chunksize=2), 1):
            if "error" in r:
                raise SystemExit(f"{r['case']}: {r['error']}")
            cases.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    mins = [float(x) for x in args.min_volumes.split(",")]
    overs = [float(x) for x in args.overlaps.split(",")]
    table, rows = {}, []
    for m in mins:
        for o in overs:
            s = score(cases, m, o)
            table[f"{m}|{o}"] = s
            rows.append((m, o, s))

    base = table[f"{mins[0]}|{max(overs)}"] if mins[0] == 0 else score(cases, 0, 1.01)
    best_key, best = max(table.items(), key=lambda kv: kv[1]["dice"])

    print(f"\n{'min mL':>7s} {'ovlp':>6s} {'Dice':>8s} {'median':>8s} "
          f"{'FP mL':>8s} {'FN mL':>8s} {'neg+FP':>7s} {'empty':>6s}")
    print("-" * 64)
    for m, o, s in rows:
        star = " <-" if f"{m}|{o}" == best_key else ""
        print(f"{m:>7.1f} {o:>6.2f} {s['dice']:>8.4f} {s['dice_median']:>8.4f} "
              f"{s['fp_ml']:>8.2f} {s['fn_ml']:>8.2f} {s['neg_with_fp']:>7d} "
              f"{s['empty_positive']:>6d}{star}")

    print(f"\nraw            : Dice {base['dice']:.4f}  FP {base['fp_ml']:.2f} mL  "
          f"neg-with-FP {base['neg_with_fp']}")
    print(f"best by Dice   : {best_key}  Dice {best['dice']:.4f} "
          f"({best['dice'] - base['dice']:+.4f})  FP {best['fp_ml']:.2f} mL "
          f"({best['fp_ml'] - base['fp_ml']:+.2f})  neg-with-FP {best['neg_with_fp']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"table": table, "best": best_key, "raw": base, "allowed_organs": organs,
         "n_cases": len(cases)}, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
