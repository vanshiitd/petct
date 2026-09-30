#!/usr/bin/env python3
"""Put predictions made on the MAE-prepared grid back onto the original scan.

Dataset503 is body-cropped, so a prediction comes out in that cropped grid. To
score it against the original `tumorSeg.nii.gz` -- and so against the same
ground truth the Dataset501 baseline was scored on -- it has to go back:

    undo any resampling (nearest neighbour, labels)
        -> paste into the crop box inside an all-zero volume of the original shape
        -> save with the original scan's geometry

For this dataset the resampling step is the identity (the source spacing rounds
to exactly the target), so the round trip is lossless and a prepared label mapped
back equals the original label exactly. The resampling branch is kept for data
where that does not hold.

    python scripts/map_back_predictions.py --pred <dir> \\
        --inverse <Dataset503>/inverse --out <dir>
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def map_one(job) -> dict:
    pred_path, meta_path, out_path = (Path(p) for p in job)
    meta = json.loads(meta_path.read_text())

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    prepared_shape = tuple(meta["prepared_shape_zyx"])
    if arr.shape != prepared_shape:
        return {"case": pred_path.name, "status": "shape_mismatch",
                "detail": f"prediction {arr.shape} != prepared {prepared_shape}"}

    # 1. undo the resampling, back to the shape the crop was taken from
    zoom = np.asarray(meta["zoom_zyx"], float)
    if not np.allclose(zoom, 1.0):
        # order=0: a label map must not gain values that were never in it
        target = tuple(int(round(s / z)) for s, z in zip(arr.shape, zoom))
        arr = ndi.zoom(arr, np.asarray(target) / np.asarray(arr.shape), order=0)

    # 2. paste into the original-sized volume
    full = np.zeros(tuple(meta["original_shape_zyx"]), dtype=np.uint8)
    bbox = meta["crop_bbox_zyx"]
    if bbox is None:
        if arr.shape != full.shape:
            return {"case": pred_path.name, "status": "shape_mismatch",
                    "detail": f"uncropped case but {arr.shape} != {full.shape}"}
        full = arr.astype(np.uint8)
    else:
        (z0, z1), (y0, y1), (x0, x1) = bbox
        region = full[z0:z1, y0:y1, x0:x1]
        if region.shape != arr.shape:
            return {"case": pred_path.name, "status": "shape_mismatch",
                    "detail": f"crop box {region.shape} != prediction {arr.shape}"}
        full[z0:z1, y0:y1, x0:x1] = arr.astype(np.uint8)

    # 3. restore the original geometry
    ref = sitk.ReadImage(meta["source_pet"])
    out = sitk.GetImageFromArray(full)
    out.CopyInformation(ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))
    return {"case": pred_path.name, "status": "ok", "voxels": int(full.sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True, help="predictions in the prepared grid")
    p.add_argument("--inverse", type=Path, required=True, help="Dataset503/inverse")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()

    preds = sorted(args.pred.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No *.nii.gz under {args.pred}")

    jobs, missing = [], []
    for f in preds:
        case = f.name[: -len(".nii.gz")]
        meta = args.inverse / f"{case}.json"
        if not meta.exists():
            missing.append(case)
            continue
        jobs.append((str(f), str(meta), str(args.out / f.name)))
    if missing:
        print(f"WARNING: {len(missing)} prediction(s) have no inverse metadata: {missing[:5]}")

    print(f"mapping {len(jobs)} predictions back to their original grids")
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(map_one, jobs, chunksize=4), 1):
            results.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")

    bad = [r for r in results if r["status"] != "ok"]
    print(f"\nmapped {len(results) - len(bad)} | failed {len(bad)}")
    for r in bad[:10]:
        print(f"   {r['case']}: {r['detail']}")
    print(f"\nwritten to {args.out}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
