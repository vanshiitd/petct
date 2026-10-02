#!/usr/bin/env python3
"""Independently re-read the DICOM-SEG and compare it with our tumorSeg.nii.gz.

Our converter reads the SEG with SimpleITK and, when the sizes happen to match,
copies the PET's geometry onto it (`CopyInformation`) rather than resampling.
That is only correct if the SEG really is on the PET grid. If it is not -- a
different frame order, a z flip, an origin offset, or sparse frames covering
only part of the volume -- the mask would be silently misplaced, and every model
trained on it would be learning shifted labels.

So this rebuilds the mask from first principles: each frame of the multi-frame
SEG is placed by its own ImagePositionPatient, mapped into the PET's index space
through the PET's own geometry, and compared voxel for voxel with what we
produced.

Also reports what the SEG references (PET or CT series) and whether its frames
are full or sparse, since both bear on whether the simple path was ever valid.

    python scripts/check_seg_conversion.py --cases cases.txt --limit 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def scan_dirs(case_id: str, nifti_root: Path, raw_root: Path):
    direct = nifti_root / case_id
    if (direct / "PET.nii.gz").exists():
        return direct, raw_root / case_id
    parts = case_id.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    rel = case_id[len(patient) + 1:]
    return nifti_root / patient / rel, raw_root / patient / rel


def read_seg_frames(seg_path: Path):
    """Mask frames plus their patient-space positions, read frame by frame."""
    ds = pydicom.dcmread(str(seg_path))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    pffg = ds.get("PerFrameFunctionalGroupsSequence", None)
    if pffg is None:
        return None

    positions, seg_numbers = [], []
    for fg in pffg:
        pos = fg.PlanePositionSequence[0].ImagePositionPatient
        positions.append([float(v) for v in pos])
        seg_id = fg.get("SegmentIdentificationSequence", None)
        seg_numbers.append(int(seg_id[0].ReferencedSegmentNumber) if seg_id else 1)

    shared = ds.get("SharedFunctionalGroupsSequence", [None])[0]
    orient = None
    if shared is not None and "PlaneOrientationSequence" in shared:
        orient = [float(v) for v in shared.PlaneOrientationSequence[0].ImageOrientationPatient]

    # what does this SEG say it was drawn on?
    ref_series = []
    for key in ("ReferencedSeriesSequence", "ReferencedSeriesSequence"):
        for rs in ds.get(key, []) or []:
            ref_series.append(str(rs.get("SeriesInstanceUID", "")))
    ref_sop_classes = set()
    for rs in ds.get("ReferencedSeriesSequence", []) or []:
        for inst in rs.get("ReferencedInstanceSequence", []) or []:
            ref_sop_classes.add(str(inst.get("ReferencedSOPClassUID", "")))

    return {
        "frames": arr, "positions": np.array(positions), "orientation": orient,
        "n_frames": len(positions), "segments": sorted(set(seg_numbers)),
        "referenced_series": ref_series, "referenced_sop_classes": sorted(ref_sop_classes),
        "rows": int(ds.Rows), "cols": int(ds.Columns),
    }


SOP_NAMES = {
    "1.2.840.10008.5.1.4.1.1.128": "PET",
    "1.2.840.10008.5.1.4.1.1.130": "PET (enhanced)",
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.2.1": "CT (enhanced)",
}


def check_case(case_id: str, nifti_root: Path, raw_root: Path, tmp_root: Path) -> dict:
    nifti_dir, raw_dir = scan_dirs(case_id, nifti_root, raw_root)
    seg_zip = raw_dir / "SEG.zip"
    if not seg_zip.exists():
        return {"case": case_id, "status": "no_seg_zip"}

    work = Path(tempfile.mkdtemp(prefix="segchk_", dir=tmp_root))
    try:
        with zipfile.ZipFile(seg_zip) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
            if not names:
                return {"case": case_id, "status": "no_dcm"}
            z.extract(names[0], work)
            info = read_seg_frames(work / names[0])
        if info is None:
            return {"case": case_id, "status": "no_per_frame_groups"}

        pet = sitk.ReadImage(str(nifti_dir / "PET.nii.gz"))
        ours = sitk.GetArrayFromImage(sitk.ReadImage(str(nifti_dir / "tumorSeg.nii.gz"))) > 0
        nz, ny, nx = ours.shape

        # place every frame by its own patient-space position
        rebuilt = np.zeros(ours.shape, dtype=bool)
        placed, out_of_range = 0, 0
        for i, pos in enumerate(info["positions"]):
            idx = pet.TransformPhysicalPointToIndex([float(v) for v in pos])
            z = int(idx[2])
            if 0 <= z < nz:
                frame = info["frames"][i].astype(bool)
                if frame.shape == (ny, nx):
                    rebuilt[z] |= frame
                    placed += 1
                else:
                    out_of_range += 1
            else:
                out_of_range += 1

        inter = int(np.logical_and(rebuilt, ours).sum())
        denom = int(rebuilt.sum()) + int(ours.sum())
        dice = 1.0 if denom == 0 else 2.0 * inter / denom

        # if they disagree, is it a constant shift along z?
        best_shift, best_dice = 0, dice
        if dice < 0.999 and rebuilt.any() and ours.any():
            for s in range(-6, 7):
                if s == 0:
                    continue
                shifted = np.roll(rebuilt, s, axis=0)
                d2 = 2 * int(np.logical_and(shifted, ours).sum()) / max(1, int(shifted.sum()) + int(ours.sum()))
                if d2 > best_dice:
                    best_dice, best_shift = d2, s
        flip_dice = None
        if dice < 0.999 and rebuilt.any() and ours.any():
            fl = rebuilt[::-1]
            flip_dice = 2 * int(np.logical_and(fl, ours).sum()) / max(1, int(fl.sum()) + int(ours.sum()))

        sop = [SOP_NAMES.get(s, s) for s in info["referenced_sop_classes"]]
        return {
            "case": case_id, "status": "ok", "dice": dice,
            "n_frames": info["n_frames"], "pet_slices": nz,
            "frames_placed": placed, "frames_out_of_range": out_of_range,
            "sparse": info["n_frames"] < nz,
            "segments": info["segments"],
            "referenced": sop or ["(none listed)"],
            "voxels_ours": int(ours.sum()), "voxels_rebuilt": int(rebuilt.sum()),
            "best_z_shift": best_shift, "dice_at_best_shift": best_dice,
            "dice_if_z_flipped": flip_dice,
        }
    except Exception as e:
        return {"case": case_id, "status": "error", "detail": repr(e)[:200]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--nifti-root", type=Path, default=Path(r"D:\data\autopet_nifti"))
    p.add_argument("--raw-root", type=Path, default=Path(r"D:\data\dataset"))
    p.add_argument("--tmp-dir", type=Path, default=Path(r"D:\data\convert_tmp"))
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path(r"D:\data\reports\seg_conversion_check.json"))
    args = p.parse_args()

    cases = [c.strip() for c in args.cases.read_text().splitlines() if c.strip()][: args.limit]
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(cases, 1):
        r = check_case(c, args.nifti_root, args.raw_root, args.tmp_dir)
        rows.append(r)
        if r["status"] == "ok":
            print(f"[{i}/{len(cases)}] {c[:34]:34s} Dice {r['dice']:.4f} | frames {r['n_frames']}"
                  f"/{r['pet_slices']} {'SPARSE' if r['sparse'] else 'full'} | "
                  f"ours {r['voxels_ours']:6d} vs rebuilt {r['voxels_rebuilt']:6d}", flush=True)
        else:
            print(f"[{i}/{len(cases)}] {c}: {r['status']} {r.get('detail','')}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        d = np.array([r["dice"] for r in ok])
        print(f"\n=== {len(ok)} scans checked ===")
        print(f"Dice vs our tumorSeg: min {d.min():.4f} | median {np.median(d):.4f} | "
              f"mean {d.mean():.4f} | == 1.0: {(d >= 0.9999).sum()}")
        shifts = [r["best_z_shift"] for r in ok if r["dice"] < 0.999]
        if shifts:
            print(f"scans below 0.999: {len(shifts)}; best z shifts {sorted(set(shifts))}")
        else:
            print("no scan needed a z shift -- no constant offset or flip")
        refs = {}
        for r in ok:
            refs[tuple(r["referenced"])] = refs.get(tuple(r["referenced"]), 0) + 1
        print(f"SEG references: {dict(refs)}")
        sparse = sum(1 for r in ok if r["sparse"])
        print(f"sparse frame sets (fewer frames than PET slices): {sparse} of {len(ok)}")
        print(f"segment numbers seen: {sorted({s for r in ok for s in r['segments']})}")
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
