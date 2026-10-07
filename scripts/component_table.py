#!/usr/bin/env python3
"""One row per predicted component and per true lesion: Phase II Part D.

This is the input to the Phase III reasoning module, and it is also what Part
C3 analyses, so it is built once and used twice. Rows are of two kinds:

  * `pred` -- a connected component of the ensemble segmentation (P_bar > 0.5),
    labelled TP if it overlaps any true lesion and FP otherwise.
  * `true` -- a connected component of the ground truth, labelled `hit` if any
    predicted voxel lands on it and `missed` otherwise.

A missed lesion has no predicted component to attach uncertainty to, so its
uncertainty is measured over the lesion's own extent: the question for Part C3
is whether the model was at least *unsure* where it failed to predict, which is
what would make the miss recoverable.

Everything is computed in the prepared grid, where the uncertainty maps and
`labels*_prep` already live. Two quantities have to come from outside it:

  * **SUV** -- the prepared PET channel is per-scan z-scored, so it is not SUV
    and SUVmax from it would be meaningless. The original `PET.nii.gz` from the
    v2 conversion is in SUV, and the crop recorded in the inverse metadata has
    zoom 1.0, so slicing it into the prepared grid is exact.
  * **organ** -- TotalSegmentator masks live in the original grid too. They are
    checked against the prepared geometry by shape after the same crop, and a
    case without a mask gets an empty organ rather than a wrong one. Only 9 of
    the 50 fit-check cases have organ masks, so that column is largely absent
    for the training rows; val and test are fully covered.

SUVpeak is the mean over a 1 mL sphere centred on the component's hottest
voxel, which is the standard definition and much less noisy than SUVmax.

    python scripts/component_table.py --maps D:\\data\\uncertainty\\val \\
        --out D:\\data\\reports\\components_val.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
TOTALSEG = Path(r"D:\data\totalseg_ml_v2")
ORGANS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 5: "liver",
          21: "urinary_bladder", 51: "heart", 90: "brain"}
U_KEYS = ("u_ent", "u_epi", "u_stab")


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(f"no prepared label for {case}")


def cropped(path: Path, bbox, shape, name: str, case: str) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    arr = arr[z0:z1, y0:y1, x0:x1]
    if arr.shape != shape:
        raise SystemExit(f"{case}: cropped {name} {arr.shape} != prepared {shape}")
    return arr


def sphere_offsets(spacing, volume_ml: float = 1.0) -> np.ndarray:
    """Voxel offsets inside a sphere of the given volume, for SUVpeak."""
    r = (3.0 * volume_ml * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)   # mm
    rz, ry, rx = (int(np.ceil(r / s)) for s in spacing)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    d2 = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2
    keep = d2 <= r * r
    return np.stack([zz[keep], yy[keep], xx[keep]], axis=1)


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

    rows = []
    no_organ = []
    for i, f in enumerate(files, 1):
        case = f.stem
        z = np.load(f)
        p_bar = z["p_bar"].astype(np.float32)
        u = {k: z[k].astype(np.float32) for k in U_KEYS}
        spacing = tuple(float(v) for v in z["spacing"])
        voxel_ml = float(np.prod(spacing)) / 1000.0
        shape = p_bar.shape

        meta = json.loads((INV / f"{case}.json").read_text())
        bbox = meta["crop_bbox_zyx"]
        suv = cropped(Path(meta["source_pet"]), bbox, shape, "PET", case).astype(np.float32)

        ts_file = TOTALSEG / f"{case}.nii.gz"
        if ts_file.exists():
            organ = cropped(ts_file, bbox, shape, "organ mask", case)
        else:
            organ = None
            no_organ.append(case)

        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
        pred = p_bar > 0.5
        p_lab, p_n = ndi.label(pred)
        g_lab, g_n = ndi.label(gt)
        offs = sphere_offsets(spacing)

        def features(mask, kind, cid, status):
            idx = np.flatnonzero(mask.ravel())
            vals = suv.ravel()[idx]
            hottest = np.unravel_index(idx[int(np.argmax(vals))], shape)
            pts = np.array(hottest) + offs
            ok = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
            peak = float(suv[tuple(pts[ok].T)].mean()) if ok.any() else float(vals.max())
            cz, cy, cx = ndi.center_of_mass(mask)
            row = {"case": case, "kind": kind, "component_id": cid, "status": status,
                   "volume_ml": float(mask.sum() * voxel_ml),
                   "suv_max": float(vals.max()), "suv_mean": float(vals.mean()),
                   "suv_peak_1ml": peak,
                   "centroid_z": round(float(cz), 2), "centroid_y": round(float(cy), 2),
                   "centroid_x": round(float(cx), 2),
                   "n_pred_in_case": p_n, "n_true_in_case": g_n,
                   "case_gt_positive": int(gt.any())}
            for k in U_KEYS:
                row[f"{k}_mean"] = float(u[k][mask].mean())
                row[f"{k}_max"] = float(u[k][mask].max())
            row["p_bar_mean"] = float(p_bar[mask].mean())
            row["p_bar_max"] = float(p_bar[mask].max())
            if organ is None:
                row["organ"] = ""
                row["organ_fraction"] = ""
            else:
                ids, counts = np.unique(organ[mask], return_counts=True)
                keep = ids != 0
                if keep.any():
                    j = int(np.argmax(counts[keep]))
                    row["organ"] = ORGANS.get(int(ids[keep][j]), f"id{int(ids[keep][j])}")
                    row["organ_fraction"] = round(float(counts[keep][j] / mask.sum()), 4)
                else:
                    row["organ"] = "none"
                    row["organ_fraction"] = 0.0
            return row

        if p_n:
            hit_by_pred = np.bincount(p_lab[gt].ravel(), minlength=p_n + 1) if gt.any() \
                else np.zeros(p_n + 1, dtype=np.int64)
            for cid in range(1, p_n + 1):
                rows.append(features(p_lab == cid, "pred", cid,
                                     "TP" if hit_by_pred[cid] > 0 else "FP"))
        if g_n:
            hit_gt = np.bincount(g_lab[pred].ravel(), minlength=g_n + 1) if pred.any() \
                else np.zeros(g_n + 1, dtype=np.int64)
            for cid in range(1, g_n + 1):
                rows.append(features(g_lab == cid, "true", cid,
                                     "hit" if hit_gt[cid] > 0 else "missed"))
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} | {len(rows)} rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"  predicted components: TP {c['TP']}, FP {c['FP']}")
    print(f"  true lesions        : hit {c['hit']}, missed {c['missed']}")
    if no_organ:
        print(f"  no organ mask for {len(no_organ)} case(s); organ left empty, not guessed")


if __name__ == "__main__":
    main()
