#!/usr/bin/env python3
r"""Phase XIII: context features for each predicted component.

See `reports/phase13_context_detector_plan.md`. The current detector sees a
component almost in isolation; these add what surrounds it -- local contrast,
edge sharpness, texture, neighbouring anatomy, left-right symmetry and its
rank within the patient. Everything is computed in the prepared grid, with
PET, CT and organ masks sliced through the recorded crop exactly as in
`build_candidates.py`. Components are M0's `P > 0.5` 6-connected labels, so
`component_id` matches the candidate table.

Writes a new table: the candidate table's columns plus the new ones.

    python scripts/phase13_context_features.py --maps D:\data\cv\oof \
        --cand D:\data\reports\cand_cv_oof_pos.csv \
        --out D:\data\reports\cand_cv_oof_ctx.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from build_candidates import INV, TOTALSEG, PHYSIO, LIVER, crop_like

NEW = ["shell_suv_mean", "tbr_mean", "tbr_max", "edge_contrast",
       "suv_cv", "frac_hot50", "hu_std", "frac_bone_hu", "frac_air_hu",
       "shell_frac_air", "shell_frac_bone", "shell_frac_physio",
       "shell_frac_liver", "p_frac90", "p_shell_mean", "uent_shell_mean",
       "mirror_suv_ratio", "suvmax_rank_case", "suvmax_over_case_p95",
       "n_pred_case", "dist_nearest_pred_mm"]
SHELL_MM = (2.0, 10.0)


def ball(spacing, r_mm):
    rz, ry, rx = (int(np.ceil(r_mm / s)) for s in spacing)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return ((zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2
            + (xx * spacing[2]) ** 2) <= r_mm * r_mm


def safe(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else float("nan")


def case_features(f: Path, cids: list[int]) -> dict[int, dict]:
    case = f.stem
    z = np.load(f)
    P = z["p_bar"].astype(np.float32)
    U = z["u_ent"].astype(np.float32)
    spacing = tuple(float(v) for v in z["spacing"])
    shape = P.shape
    meta = json.loads((INV / f"{case}.json").read_text())
    bbox = meta["crop_bbox_zyx"]
    suv = crop_like(Path(meta["source_pet"]), bbox, shape, "PET", case).astype(np.float32)
    ct = crop_like(Path(meta["source_pet"]).with_name("CT_resample.nii.gz"),
                   bbox, shape, "CT", case).astype(np.float32)
    ts = TOTALSEG / f"{case}.nii.gz"
    organ = crop_like(ts, bbox, shape, "organ", case) if ts.exists() else None

    m0 = P > 0.5
    lab, _ = ndi.label(m0)
    body = ct > -500
    p95 = float(np.percentile(suv[body], 95)) if body.any() else float("nan")
    b_in, b_out = ball(spacing, SHELL_MM[0]), ball(spacing, SHELL_MM[1])
    one = ndi.generate_binary_structure(3, 1)

    objs = ndi.find_objects(lab)
    pad = [int(np.ceil(SHELL_MM[1] / s)) + 1 for s in spacing]
    cent, smax, out = {}, {}, {}
    for cid in cids:
        sl = objs[cid - 1]
        box = tuple(slice(max(0, s.start - p), min(n, s.stop + p))
                    for s, p, n in zip(sl, pad, shape))
        m = lab[box] == cid
        other = m0[box] & ~m
        sv, hu, pp, uu = suv[box], ct[box], P[box], U[box]
        shell = ndi.binary_dilation(m, b_out) & ~ndi.binary_dilation(m, b_in) & ~other
        inner = m & ~ndi.binary_erosion(m, one)
        ring = ndi.binary_dilation(m, one) & ~m & ~other
        vin = sv[m]
        mx, mean = float(vin.max()), float(vin.mean())
        sh = float(sv[shell].mean()) if shell.any() else float("nan")
        hin = hu[m]
        r = {"shell_suv_mean": sh, "tbr_mean": safe(mean, sh), "tbr_max": safe(mx, sh),
             "edge_contrast": (float((sv[inner].mean() - sv[ring].mean()) / mx)
                               if ring.any() and mx > 0 else float("nan")),
             "suv_cv": safe(float(vin.std()), mean),
             "frac_hot50": float((vin >= 0.5 * mx).mean()),
             "hu_std": float(hin.std()),
             "frac_bone_hu": float((hin > 200).mean()),
             "frac_air_hu": float((hin < -400).mean()),
             "shell_frac_air": float((hu[shell] < -400).mean()) if shell.any() else float("nan"),
             "shell_frac_bone": float((hu[shell] > 200).mean()) if shell.any() else float("nan"),
             "p_frac90": float((pp[m] > 0.9).mean()),
             "p_shell_mean": float(pp[shell].mean()) if shell.any() else float("nan"),
             "uent_shell_mean": float(uu[shell].mean()) if shell.any() else float("nan"),
             "suvmax_over_case_p95": safe(mx, p95), "_n": int(m.sum())}
        if organ is not None and shell.any():
            og = organ[box][shell]
            r["shell_frac_physio"] = float(np.isin(og, PHYSIO).mean())
            r["shell_frac_liver"] = float((og == LIVER).mean())
        else:
            r["shell_frac_physio"] = r["shell_frac_liver"] = float("nan")
        # left-right mirror about the body midline of the component's slices
        pts = np.argwhere(m) + np.array([b.start for b in box])
        zs = np.unique(pts[:, 0])
        bx = np.flatnonzero(body[zs].any(axis=(0, 1)))
        if bx.size:
            mid2 = bx[0] + bx[-1]
            xm = mid2 - pts[:, 2]
            ok = (xm >= 0) & (xm < shape[2])
            r["mirror_suv_ratio"] = (safe(float(suv[pts[ok, 0], pts[ok, 1], xm[ok]].mean()), mean)
                                     if ok.mean() >= 0.5 else float("nan"))
        else:
            r["mirror_suv_ratio"] = float("nan")
        cent[cid] = pts.mean(0) * np.array(spacing)
        smax[cid] = mx
        out[cid] = r

    vals = np.array([smax[c] for c in cids])
    C = np.array([cent[c] for c in cids])
    for i, cid in enumerate(cids):
        out[cid]["n_pred_case"] = len(cids)
        out[cid]["suvmax_rank_case"] = (float((vals < vals[i]).sum() / (len(cids) - 1))
                                        if len(cids) > 1 else float("nan"))
        if len(cids) > 1:
            d = np.linalg.norm(C - C[i], axis=1)
            d[i] = np.inf
            out[cid]["dist_nearest_pred_mm"] = float(d.min())
        else:
            out[cid]["dist_nearest_pred_mm"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.cand, newline="", encoding="utf-8")))
    cols = list(rows[0].keys()) + NEW
    by_case: dict[str, list[int]] = {}
    for r in rows:
        if r["kind"] == "pred":
            by_case.setdefault(r["case"], []).append(int(r["component_id"]))
    cases = sorted(by_case)[: args.limit] if args.limit else sorted(by_case)
    print(f"{len(cases)} cases with predicted components", flush=True)

    feats: dict[tuple[str, int], dict] = {}
    nvox = {(r["case"], int(r["component_id"])): int(r["n_voxels"])
            for r in rows if r["kind"] == "pred"}
    for i, case in enumerate(cases, 1):
        for cid, r in case_features(args.maps / f"{case}.npz", sorted(by_case[case])).items():
            # the component must be the candidate table's, voxel for voxel
            if r.pop("_n") != nvox[(case, cid)]:
                raise SystemExit(f"{case} component {cid}: size differs from the table")
            feats[(case, cid)] = r
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            if args.limit and r["case"] not in by_case.keys() & set(cases):
                continue
            extra = feats.get((r["case"], int(r["component_id"]))) if r["kind"] == "pred" else None
            w.writerow({**r, **{k: (extra[k] if extra else "") for k in NEW}})
    miss = {k: sum(not np.isfinite(v[k]) for v in feats.values()) for k in NEW}
    print(f"written {args.out}: {len(feats)} components; missing per feature "
          f"{ {k: n for k, n in miss.items() if n} }")


if __name__ == "__main__":
    main()
