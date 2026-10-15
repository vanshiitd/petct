#!/usr/bin/env python3
"""Candidate units and their features, for Experiments 3 and 4.

Two kinds of candidate, both from M0 (the Dataset505 MAE-pretrained model alone,
8 flips, no second model):

  * **pred** -- a connected component of `P_M0 > 0.5`, i.e. a thing M0 actually
    predicts. Labelled TP if it touches any ground-truth lesion, else FP. These
    are what action A may remove.
  * **recover** -- a connected component of `P_M0 > t_low` that does *not* touch
    M0's own output. These are places M0 nearly fired and did not. Labelled
    `recoverable` if it touches a ground-truth lesion M0 missed, else `spurious`.
    These are what action B may add.

Connectivity is scipy's default 6-connectivity everywhere, matching
`evaluate_predictions.py`, so component counts here and in the evaluator mean
the same thing. Components below `--min-voxels` are dropped before anything
else: at t_low = 0.1 a whole-body probability map breaks into tens of thousands
of single-voxel specks, which are not candidates for any action and would
dominate every statistic. The number dropped is reported rather than hidden.

Features are grouped so Experiment 3 can switch groups on and off:

  model      mean/max P, mean/max U_ent, mean/max U_stab_single
  metabolic  SUVmax/mean/peak(1 mL sphere), and each over the patient's liver
             SUVmean -- the standard PET reference, so a lesion is judged
             against that patient's own physiology rather than an absolute
  anatomy    dominant TotalSegmentator organ, fraction inside physiological
             uptake organs (brain, heart, bladder, kidneys), liver and spleen
             fractions kept separate because they hold real lesion volume,
             and distance to the nearest physiological-uptake organ
  ct         mean/median HU, fraction in the fat range (-190..-30 HU, where
             brown fat sits) and the fluid range (-10..30 HU, where urine sits)
  shape      volume, sphericity, bounding-box extents, elongation, component
             count in the patient
  position   centroid z as a fraction of the body extent

Everything is computed in the prepared grid; SUV, CT and organ masks are sliced
from the originals through the recorded crop, which has zoom 1.0 and is exact.

    python scripts/build_candidates.py --maps D:\\data\\uncertainty_m0\\val \\
        --out D:\\data\\reports\\cand_val.csv --t-low 0.1
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
PHYSIO = [2, 3, 21, 51, 90]        # kidneys, bladder, heart, brain
LIVER, SPLEEN = 5, 1


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(case)


def crop_like(path: Path, bbox, shape, what: str, case: str) -> np.ndarray:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    a = a[z0:z1, y0:y1, x0:x1]
    if a.shape != shape:
        raise SystemExit(f"{case}: cropped {what} {a.shape} != prepared {shape}")
    return a


def sphere_offsets(spacing, volume_ml: float = 1.0) -> np.ndarray:
    r = (3.0 * volume_ml * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    rz, ry, rx = (int(np.ceil(r / s)) for s in spacing)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    d2 = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2
    k = d2 <= r * r
    return np.stack([zz[k], yy[k], xx[k]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t-low", type=float, default=0.1)
    ap.add_argument("--min-voxels", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(args.maps.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} cases | t_low {args.t_low} | min {args.min_voxels} voxels",
          flush=True)

    rows, dropped, no_organ = [], 0, []
    for i, f in enumerate(files, 1):
        case = f.stem
        z = np.load(f)
        P = z["p_bar"].astype(np.float32)
        u_ent = z["u_ent"].astype(np.float32)
        u_stab = z["u_stab"].astype(np.float32)
        spacing = tuple(float(v) for v in z["spacing"])
        voxel_ml = float(np.prod(spacing)) / 1000.0
        shape = P.shape

        meta = json.loads((INV / f"{case}.json").read_text())
        bbox = meta["crop_bbox_zyx"]
        suv = crop_like(Path(meta["source_pet"]), bbox, shape, "PET", case).astype(np.float32)
        ct = crop_like(Path(meta["source_pet"]).with_name("CT_resample.nii.gz"),
                       bbox, shape, "CT", case).astype(np.float32)
        ts_file = TOTALSEG / f"{case}.nii.gz"
        organ = crop_like(ts_file, bbox, shape, "organ", case) if ts_file.exists() else None
        if organ is None:
            no_organ.append(case)

        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
        m0 = P > 0.5
        body = ct > -500
        zs = np.flatnonzero(body.any(axis=(1, 2)))
        z_lo, z_hi = (int(zs[0]), int(zs[-1])) if zs.size else (0, shape[0] - 1)

        liver_suv = float(suv[organ == LIVER].mean()) if (
            organ is not None and (organ == LIVER).any()) else float("nan")
        physio = np.isin(organ, PHYSIO) if organ is not None else None
        dist_physio = (ndi.distance_transform_edt(~physio, sampling=spacing)
                       if physio is not None and physio.any() else None)
        offs = sphere_offsets(spacing)

        g_lab, g_n = ndi.label(gt)
        missed_gt = np.zeros_like(gt)
        if g_n:
            touched = np.bincount(g_lab[m0].ravel(), minlength=g_n + 1) if m0.any() \
                else np.zeros(g_n + 1, dtype=np.int64)
            missed_ids = np.flatnonzero(touched[1:] == 0) + 1
            if missed_ids.size:
                missed_gt = np.isin(g_lab, missed_ids)

        def emit(mask, kind, cid, status):
            idx = np.flatnonzero(mask.ravel())
            sv = suv.ravel()[idx]
            hot = np.unravel_index(idx[int(np.argmax(sv))], shape)
            pts = np.array(hot) + offs
            ok = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
            peak = float(suv[tuple(pts[ok].T)].mean()) if ok.any() else float(sv.max())
            hu = ct[mask]
            cz, cy, cx = ndi.center_of_mass(mask)
            n = int(mask.sum())
            vol = n * voxel_ml
            pos = np.argwhere(mask)
            ext = (pos.max(axis=0) - pos.min(axis=0) + 1) * np.array(spacing)
            r_eq = (3.0 * vol * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
            r_bb = float(max(ext)) / 2.0
            row = {
                "case": case, "kind": kind, "component_id": cid, "status": status,
                "case_gt_positive": int(gt.any()),
                # model
                "p_mean": float(P[mask].mean()), "p_max": float(P[mask].max()),
                "uent_mean": float(u_ent[mask].mean()), "uent_max": float(u_ent[mask].max()),
                "ustab_mean": float(u_stab[mask].mean()), "ustab_max": float(u_stab[mask].max()),
                # metabolic
                "suv_max": float(sv.max()), "suv_mean": float(sv.mean()),
                "suv_peak": peak,
                "suv_max_over_liver": float(sv.max() / liver_suv) if liver_suv == liver_suv
                and liver_suv > 0 else float("nan"),
                "suv_peak_over_liver": float(peak / liver_suv) if liver_suv == liver_suv
                and liver_suv > 0 else float("nan"),
                "liver_suv_mean": liver_suv,
                # ct
                "hu_mean": float(hu.mean()), "hu_median": float(np.median(hu)),
                "frac_fat_hu": float(((hu >= -190) & (hu <= -30)).mean()),
                "frac_fluid_hu": float(((hu >= -10) & (hu <= 30)).mean()),
                # shape / position
                "volume_ml": vol, "n_voxels": n,
                "sphericity": float(r_eq / r_bb) if r_bb > 0 else float("nan"),
                "extent_z": float(ext[0]), "extent_y": float(ext[1]), "extent_x": float(ext[2]),
                "elongation": float(max(ext) / max(min(ext), 1e-6)),
                "centroid_z": round(float(cz), 2),
                "z_rel": float((cz - z_lo) / max(1, z_hi - z_lo)),
            }
            if organ is None:
                row.update({"organ": "", "organ_fraction": "", "frac_physio": "",
                            "frac_liver": "", "frac_spleen": "", "dist_physio_mm": ""})
            else:
                ids, cnt = np.unique(organ[mask], return_counts=True)
                keep = ids != 0
                if keep.any():
                    j = int(np.argmax(cnt[keep]))
                    row["organ"] = ORGANS.get(int(ids[keep][j]), f"id{int(ids[keep][j])}")
                    row["organ_fraction"] = round(float(cnt[keep][j] / n), 4)
                else:
                    row["organ"], row["organ_fraction"] = "none", 0.0
                row["frac_physio"] = float(physio[mask].mean())
                row["frac_liver"] = float((organ[mask] == LIVER).mean())
                row["frac_spleen"] = float((organ[mask] == SPLEEN).mean())
                row["dist_physio_mm"] = (float(dist_physio[mask].min())
                                         if dist_physio is not None else float("nan"))
            return row

        # --- predicted components -------------------------------------
        p_lab, p_n = ndi.label(m0)
        n_pred_kept = 0
        if p_n:
            sizes = np.bincount(p_lab.ravel(), minlength=p_n + 1)
            touch = np.bincount(p_lab[gt].ravel(), minlength=p_n + 1) if gt.any() \
                else np.zeros(p_n + 1, dtype=np.int64)
            for cid in range(1, p_n + 1):
                if sizes[cid] < args.min_voxels:
                    dropped += 1
                    continue
                rows.append(emit(p_lab == cid, "pred", cid,
                                 "TP" if touch[cid] > 0 else "FP"))
                n_pred_kept += 1

        # --- recovery candidates --------------------------------------
        low = (P > args.t_low) & ~m0
        # a component of the low map that touches M0 is just M0's own margin,
        # not a new finding, so it is excluded by construction
        if low.any():
            l_lab, l_n = ndi.label(low)
            sizes = np.bincount(l_lab.ravel(), minlength=l_n + 1)
            touch_m0 = np.bincount(l_lab[ndi.binary_dilation(m0)].ravel(),
                                   minlength=l_n + 1) if m0.any() \
                else np.zeros(l_n + 1, dtype=np.int64)
            touch_miss = np.bincount(l_lab[missed_gt].ravel(), minlength=l_n + 1) \
                if missed_gt.any() else np.zeros(l_n + 1, dtype=np.int64)
            for cid in range(1, l_n + 1):
                if sizes[cid] < args.min_voxels or touch_m0[cid] > 0:
                    dropped += 1
                    continue
                rows.append(emit(l_lab == cid, "recover", cid,
                                 "recoverable" if touch_miss[cid] > 0 else "spurious"))
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} | {len(rows)} rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter((r["kind"], r["status"]) for r in rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"  pred    : TP {c[('pred','TP')]}, FP {c[('pred','FP')]}")
    print(f"  recover : recoverable {c[('recover','recoverable')]}, "
          f"spurious {c[('recover','spurious')]}")
    print(f"  dropped below {args.min_voxels} voxels or touching M0: {dropped}")
    if no_organ:
        print(f"  no organ mask for {len(no_organ)} case(s); anatomy left empty")


if __name__ == "__main__":
    main()
