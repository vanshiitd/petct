#!/usr/bin/env python3
"""Validate the reconverted dataset before anything is trained on it.

The previous dataset carried mirrored lesion masks on roughly 63% of
tumour-positive scans and nothing downstream noticed, so this checks the new one
against the only arbiter that does not share the converter's assumptions: the
tracer. A lesion label sits on elevated FDG uptake; a mirrored one lands on
whatever is opposite it, which is usually unremarkable tissue.

Checks:
  1. every scan converted, and the tumour-positive set unchanged
  2. for every tumour-positive scan, mask uptake as stored against mask uptake
     flipped -- flipping must no longer help
  3. the frozen split still applies: same case ids, same tumour-positive status
     per set
  4. PET and CT agree with the old dataset up to the expected differences and,
     critically, are not themselves flipped

    python scripts/validate_v2.py --v2 C:\\data\\autopet_nifti_v2 \\
        --old D:\\data\\autopet_nifti --split splits/autopet_v1.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def scan_dir(root: Path, case_id: str) -> Path:
    d = root / case_id
    if (d / "PET.nii.gz").exists():
        return d
    parts = case_id.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    return root / patient / case_id[len(patient) + 1:]


def check_label(job) -> dict:
    case_id, v2_root = job
    d = scan_dir(Path(v2_root), case_id)
    try:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "tumorSeg.nii.gz"))) > 0
        if not seg.any():
            return {"case": case_id, "positive": False}
        pet = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "PET.nii.gz")))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "CT_resample.nii.gz")))
        body = ct > -500
        bg = max(float(np.median(pet[body & ~seg])) if (body & ~seg).any() else 1.0, 1e-6)
        as_is = float(np.median(pet[seg])) / bg
        flipped = float(np.median(pet[seg[:, ::-1, :]])) / bg
        return {"case": case_id, "positive": True, "voxels": int(seg.sum()),
                "ratio_as_is": as_is, "ratio_flipped": flipped,
                "flip_still_better": bool(flipped > as_is * 1.2)}
    except Exception as e:
        return {"case": case_id, "error": repr(e)[:160]}


def compare_channels(job) -> dict:
    """v2 against the old dataset: expected differences only, and no flip."""
    case_id, v2_root, old_root = job
    try:
        a = scan_dir(Path(v2_root), case_id)
        b = scan_dir(Path(old_root), case_id)
        out = {"case": case_id}
        for name, key in (("PET", "PET.nii.gz"), ("CT", "CT_resample.nii.gz")):
            new_img = sitk.ReadImage(str(a / key))
            old_img = sitk.ReadImage(str(b / key))
            old_on_new = sitk.Resample(old_img, new_img, sitk.Transform(),
                                       sitk.sitkLinear, -1000.0 if name == "CT" else 0.0,
                                       sitk.sitkFloat32)
            n = sitk.GetArrayFromImage(sitk.Cast(new_img, sitk.sitkFloat32))
            o = sitk.GetArrayFromImage(old_on_new)
            # correlation as stored against correlation flipped: a flip would
            # show up as the mirrored version correlating better
            def corr(x, y):
                x, y = x.ravel(), y.ravel()
                sx, sy = x.std(), y.std()
                return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan
            out[name] = {
                "corr_as_is": corr(n, o),
                "corr_flipped": corr(n, o[:, ::-1, :]),
                "max_abs_diff": float(np.abs(n - o).max()),
                "mean_abs_diff": float(np.abs(n - o).mean()),
            }
        return out
    except Exception as e:
        return {"case": case_id, "error": repr(e)[:160]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--old", type=Path, default=None)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--channel-sample", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path(r"D:\data\reports\validate_v2.json"))
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    cases = [c for s in ("train", "val", "test")
             for cs in split["splits"][s]["cases"].values() for c in cs]
    print(f"split lists {len(cases)} cases\n")

    # --- 1. completeness -------------------------------------------------
    present = [c for c in cases if (scan_dir(args.v2, c) / "tumorSeg.nii.gz").exists()]
    missing = [c for c in cases if c not in set(present)]
    print(f"1. converted: {len(present)}/{len(cases)}")
    if missing:
        print(f"   MISSING {len(missing)}: {missing[:10]}")

    # --- 2. labels land on tracer ---------------------------------------
    print("\n2. label placement (PET uptake as arbiter)")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(check_label, [(c, str(args.v2)) for c in present],
                                       chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"   {i}/{len(present)}", flush=True)
    errs = [r for r in rows if "error" in r]
    pos = [r for r in rows if r.get("positive")]
    still = [r for r in pos if r.get("flip_still_better")]
    ratios = np.array([r["ratio_as_is"] for r in pos])
    flipped = np.array([r["ratio_flipped"] for r in pos])
    print(f"   tumour-positive scans      : {len(pos)}")
    print(f"   median uptake as stored    : {np.median(ratios):.2f}x")
    print(f"   median uptake flipped      : {np.median(flipped):.2f}x")
    print(f"   flipping STILL helps on    : {len(still)} scan(s)")
    for r in still[:15]:
        print(f"      {r['case']}: as-is {r['ratio_as_is']:.2f}x vs flipped {r['ratio_flipped']:.2f}x")
    if errs:
        print(f"   errors: {len(errs)} {[e['case'] for e in errs[:5]]}")

    # --- 3. split integrity ---------------------------------------------
    print("\n3. split integrity")
    pos_now = {r["case"] for r in pos}
    ok = True
    for s in ("train", "val", "test"):
        ids = [c for cs in split["splits"][s]["cases"].values() for c in cs]
        n_pos_now = sum(1 for c in ids if c in pos_now)
        n_pos_split = split["splits"][s]["n_tumour_positive_scans"]
        flag = "OK" if n_pos_now == n_pos_split else "CHANGED"
        ok &= n_pos_now == n_pos_split
        print(f"   {s:5s}: {len(ids):3d} cases | tumour-positive now {n_pos_now:3d}, "
              f"split file says {n_pos_split:3d}  {flag}")
    print(f"   -> splits/autopet_v1.json {'remains valid as is' if ok else 'NEEDS REGENERATION'}")

    # --- 4. channels against the old dataset ----------------------------
    if args.old:
        print(f"\n4. PET/CT against the old dataset ({args.channel_sample} scans)")
        sample = present[:: max(1, len(present) // args.channel_sample)][: args.channel_sample]
        crows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for r in pool.map(compare_channels,
                              [(c, str(args.v2), str(args.old)) for c in sample], chunksize=2):
                crows.append(r)
        good = [r for r in crows if "error" not in r]
        for name in ("PET", "CT"):
            ci = np.array([r[name]["corr_as_is"] for r in good])
            cf = np.array([r[name]["corr_flipped"] for r in good])
            worse = int((cf > ci).sum())
            print(f"   {name}: corr as-is median {np.median(ci):.4f} | "
                  f"corr flipped median {np.median(cf):.4f} | "
                  f"flipped correlates better on {worse} of {len(good)} "
                  f"{'<-- FLIP SUSPECTED' if worse else '(no flip)'}")
            md = np.array([r[name]["mean_abs_diff"] for r in good])
            xd = np.array([r[name]["max_abs_diff"] for r in good])
            print(f"      mean abs diff median {np.median(md):.4f} | "
                  f"max abs diff median {np.median(xd):.1f}")
        rows += crows

    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
