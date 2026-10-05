#!/usr/bin/env python3
"""Clean up predicted lesion masks: drop tiny blobs and blobs inside organs.

Both rules target the same failure, which dominates these models: predicting
lesion on tissue that is bright for physiological reasons. Neither needs
retraining.

1. **Small-component removal.** A component below `--min-volume-ml` is dropped.
   Real lesions below a fraction of a millilitre are rare in this data, while
   speckle is common.

2. **Organ masking.** A component with more than `--organ-overlap` of its volume
   inside an allowed organ is dropped. Brain, bladder, kidneys and heart take up
   FDG physiologically, and a model that keys on "bright" fires on them.

   Which organs are allowed is NOT a judgement call: `--organ-lesion-share`
   reports what fraction of real lesion volume sits in each organ, and only
   organs holding a negligible share may be passed to `--organs`. Brain
   metastases do occur in melanoma, so that has to be measured.

Both thresholds are tuned on validation and then applied once to test.

    python scripts/postprocess.py --pred <dir> --out <dir> --min-volume-ml 0.5
    python scripts/postprocess.py --pred <dir> --out <dir> --min-volume-ml 0.5 \\
        --totalseg D:\\data\\totalseg_ml --organs brain,urinary_bladder --organ-overlap 0.5
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

# TotalSegmentator's label ids in its multilabel output, for the organs we use.
# Resolved at runtime from the installed class map when available.
DEFAULT_LABEL_IDS = {
    "spleen": 1, "kidney_right": 2, "kidney_left": 3, "liver": 5,
    "urinary_bladder": 21, "heart": 51, "brain": 90,
}


def load_label_ids(path: Path | None) -> dict[str, int]:
    if path and path.exists():
        return {k: int(v) for k, v in json.loads(path.read_text()).items()}
    return dict(DEFAULT_LABEL_IDS)


def _geom(img) -> tuple:
    """Everything that fixes where a voxel sits in the patient."""
    return (img.GetSize(),
            tuple(round(v, 6) for v in img.GetSpacing()),
            tuple(round(v, 3) for v in img.GetOrigin()),
            tuple(round(v, 6) for v in img.GetDirection()))


def clean_one(job) -> dict:
    (pred_path, out_path, min_volume_ml, totalseg_path,
     organ_ids, organ_overlap) = job
    pred_path, out_path = Path(pred_path), Path(out_path)

    img = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayFromImage(img) > 0
    sx, sy, sz = img.GetSpacing()
    voxel_ml = (sx * sy * sz) / 1000.0

    before = int(arr.sum())
    if before == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out_path))
        return {"case": pred_path.name, "status": "ok", "before": 0, "after": 0,
                "dropped_small": 0, "dropped_organ": 0}

    lab, n = ndi.label(arr)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)

    drop = np.zeros(n + 1, dtype=bool)
    dropped_small = 0
    if min_volume_ml > 0:
        too_small = (sizes * voxel_ml) < min_volume_ml
        too_small[0] = False
        drop |= too_small
        dropped_small = int(too_small.sum())

    dropped_organ = 0
    if totalseg_path and organ_ids:
        ts_file = Path(totalseg_path)
        if ts_file.exists():
            ts_img = sitk.ReadImage(str(ts_file))
            ts = sitk.GetArrayFromImage(ts_img)
            # Geometry, not shape. Organ masks built from the old conversion share
            # the prediction's (400, 400, N) shape but traverse y the other way, so
            # indexing one against the other mirrors every organ and masks the wrong
            # side of the patient -- with no shape check anywhere to notice.
            if _geom(ts_img) != _geom(img):
                return {"case": pred_path.name, "status": "geometry_mismatch",
                        "detail": f"organ mask {_geom(ts_img)} != prediction {_geom(img)}"}
            if ts.shape == arr.shape:
                organ = np.isin(ts, list(organ_ids))
                # how much of each component lies inside an allowed organ
                inside = np.bincount(lab[organ].ravel(), minlength=n + 1)
                frac = inside / np.maximum(sizes, 1)
                in_organ = frac > organ_overlap
                in_organ[0] = False
                dropped_organ = int((in_organ & ~drop).sum())
                drop |= in_organ
            else:
                return {"case": pred_path.name, "status": "shape_mismatch",
                        "detail": f"organ mask {ts.shape} != prediction {arr.shape}"}
        else:
            return {"case": pred_path.name, "status": "no_organ_mask"}

    keep = ~drop[lab]
    cleaned = (arr & keep).astype(np.uint8)

    out = sitk.GetImageFromArray(cleaned)
    out.CopyInformation(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))
    return {"case": pred_path.name, "status": "ok", "before": before,
            "after": int(cleaned.sum()), "dropped_small": dropped_small,
            "dropped_organ": dropped_organ, "n_components": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-volume-ml", type=float, default=0.0,
                   help="drop connected components smaller than this (0 disables)")
    p.add_argument("--totalseg", type=Path, default=None,
                   help="directory of TotalSegmentator multilabel masks, <case>.nii.gz")
    p.add_argument("--organs", default="",
                   help="comma-separated organ names whose contents may be dropped")
    p.add_argument("--organ-overlap", type=float, default=0.5,
                   help="drop a component with more than this fraction inside an allowed organ")
    p.add_argument("--label-ids", type=Path, default=None,
                   help="JSON mapping organ name -> TotalSegmentator label id")
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()

    label_ids = load_label_ids(args.label_ids)
    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    unknown = [o for o in organs if o not in label_ids]
    if unknown:
        raise SystemExit(f"unknown organ(s): {unknown}; known: {sorted(label_ids)}")
    organ_ids = tuple(label_ids[o] for o in organs)

    preds = sorted(args.pred.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No *.nii.gz under {args.pred}")
    print(f"{len(preds)} predictions | min volume {args.min_volume_ml} mL | "
          f"organs {organs or 'none'} at >{args.organ_overlap:.0%} overlap")

    jobs = []
    for f in preds:
        case = f.name[: -len(".nii.gz")]
        ts = str(args.totalseg / f"{case}.nii.gz") if args.totalseg else None
        jobs.append((str(f), str(args.out / f.name), args.min_volume_ml,
                     ts, organ_ids, args.organ_overlap))

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(clean_one, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")

    bad = [r for r in rows if r["status"] != "ok"]
    ok = [r for r in rows if r["status"] == "ok"]
    before = sum(r["before"] for r in ok)
    after = sum(r["after"] for r in ok)
    print(f"\ncleaned {len(ok)} | problems {len(bad)}")
    for r in bad[:10]:
        print(f"   {r['case']}: {r['status']} {r.get('detail','')}")
    print(f"voxels kept: {after:,} of {before:,} ({after/max(1,before):.1%})")
    print(f"components dropped: small {sum(r.get('dropped_small',0) for r in ok)}, "
          f"in organ {sum(r.get('dropped_organ',0) for r in ok)}")
    print(f"\nwritten to {args.out}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
