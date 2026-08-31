#!/usr/bin/env python3
"""Convert a downloaded TCIA PET/CT collection into the NIfTI layout the
training pipeline expects.

TCIA ships raw DICOM. Every other entry point in this repo expects, per scan:

    <out>/<PatientID>/PET.nii.gz
    <out>/<PatientID>/CT_resample.nii.gz
    <out>/<PatientID>/tumorSeg.nii.gz

This script bridges that gap. It auto-detects what it is given:

  * directories of .dcm files  (e.g. an NBIA Data Retriever download)
  * per-patient .zip archives  (one CT/PT/SEG archive per patient folder)
  * already-converted .nii.gz  (reported and skipped)

Series are identified by the DICOM Modality tag, not by folder name, so it does
not care how the download tool arranged things. CT is resampled onto the PET
grid so the two are voxel-aligned, and the segmentation is binarised.

Safe to interrupt and rerun: patients already converted are skipped.

Examples:
    python scripts/dicom_to_nifti.py --source /data/AutoPET_raw --target /data/autopet_nifti
    python scripts/dicom_to_nifti.py --source /data/AutoPET_raw --target /data/autopet_nifti --limit 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# DICOM tags read from the first slice of each series
TAG_PATIENT_ID = "0010|0020"
TAG_MODALITY = "0008|0060"
TAG_SERIES_UID = "0020|000e"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def read_series_header(dicom_dir: Path) -> dict | None:
    """Read PatientID / Modality from the first readable DICOM in a directory."""
    for f in sorted(dicom_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            r = sitk.ImageFileReader()
            r.SetFileName(str(f))
            r.LoadPrivateTagsOn()
            r.ReadImageInformation()
            return {
                "patient_id": (r.GetMetaData(TAG_PATIENT_ID).strip()
                               if r.HasMetaDataKey(TAG_PATIENT_ID) else None),
                "modality": (r.GetMetaData(TAG_MODALITY).strip().upper()
                             if r.HasMetaDataKey(TAG_MODALITY) else None),
                "series_uid": (r.GetMetaData(TAG_SERIES_UID).strip()
                               if r.HasMetaDataKey(TAG_SERIES_UID) else None),
                "n_files": sum(1 for x in dicom_dir.iterdir() if x.is_file()),
                "path": dicom_dir,
            }
        except Exception:
            continue  # not a DICOM, or unreadable; try the next file
    return None


def find_series(root: Path) -> list[dict]:
    """Every directory under root that holds at least one readable DICOM."""
    series = []
    candidates = {p.parent for p in root.rglob("*") if p.is_file()}
    for d in sorted(candidates):
        info = read_series_header(d)
        if info and info["modality"]:
            series.append(info)
    return series


def group_by_patient(series: list[dict], root: Path) -> dict[str, list[dict]]:
    """Group series by PatientID, falling back to the top-level folder name."""
    groups = defaultdict(list)
    for s in series:
        pid = s["patient_id"]
        if not pid:
            try:
                pid = s["path"].relative_to(root).parts[0]
            except (ValueError, IndexError):
                pid = s["path"].name
        groups[pid].append(s)
    return groups


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise RuntimeError(f"no DICOM series found in {dicom_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def load_segmentation(seg_dir: Path) -> sitk.Image:
    """DICOM-SEG is a single multi-frame file, not a slice series."""
    files = [f for f in sorted(seg_dir.iterdir()) if f.is_file()]
    last_err = None
    for f in files:
        try:
            return sitk.ReadImage(str(f))
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not read a segmentation from {seg_dir}: {last_err}")


def convert_patient(patient_id: str, series: list[dict], out_dir: Path) -> str:
    """Write PET / CT_resample / tumorSeg for one patient. Returns a status word."""
    cts = [s for s in series if s["modality"] == "CT"]
    pts = [s for s in series if s["modality"] in ("PT", "PET")]
    segs = [s for s in series if s["modality"] == "SEG"]

    missing = [n for n, v in (("CT", cts), ("PT", pts), ("SEG", segs)) if not v]
    if missing:
        print(f"  [skip] {patient_id}: missing {', '.join(missing)}")
        return "skipped"

    # several CT series can exist (e.g. different reconstructions); take the
    # one with the most slices, which is the full-resolution acquisition
    ct_dir = max(cts, key=lambda s: s["n_files"])["path"]
    pt_dir = pts[0]["path"]
    seg_dir = segs[0]["path"]

    pet = load_dicom_series(pt_dir)
    ct = load_dicom_series(ct_dir)
    seg = load_segmentation(seg_dir)

    # PET defines the reference grid; CT is resampled onto it so the two
    # channels are voxel-aligned. -1000 HU (air) is the correct fill for
    # regions outside the original CT field of view.
    ct_res = sitk.Resample(ct, pet, sitk.Transform(), sitk.sitkLinear, -1000.0, ct.GetPixelID())

    seg_bin = sitk.Cast(seg > 0, sitk.sitkUInt8)
    if seg_bin.GetSize() != pet.GetSize():
        # geometry disagrees: resample with nearest neighbour so no label
        # values are invented by interpolation
        seg_bin = sitk.Resample(seg_bin, pet, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                                seg_bin.GetPixelID())
    else:
        seg_bin.CopyInformation(pet)

    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pet, str(out_dir / "PET.nii.gz"))
    sitk.WriteImage(ct_res, str(out_dir / "CT_resample.nii.gz"))
    sitk.WriteImage(seg_bin, str(out_dir / "tumorSeg.nii.gz"))

    lesion = int(sitk.GetArrayFromImage(seg_bin).sum())
    print(f"  ok {patient_id}: {pet.GetSize()}  lesion_voxels={lesion}")
    return "ok"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="root of the downloaded collection")
    p.add_argument("--target", type=Path, required=True, help="output root for the NIfTI layout")
    p.add_argument("--limit", type=int, default=None, help="convert at most N patients (for a trial run)")
    p.add_argument("--overwrite", action="store_true", help="reconvert patients already present")
    p.add_argument("--keep-extracted", action="store_true",
                   help="keep the temporary directory used when unpacking .zip archives")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")

    # already-converted?
    existing_nii = list(args.source.rglob("PET.nii.gz"))
    if existing_nii:
        print(f"Found {len(existing_nii)} PET.nii.gz already under {args.source}.")
        print("This data appears to be converted already — point --data-root at it directly:")
        print(f"  python scripts/finetune.py --arch base --data-root {args.source} ...")
        if not list(args.source.rglob("*.dcm")):
            return

    # unpack any zips into a scratch tree first
    zips = sorted(args.source.rglob("*.zip"))
    work_root = args.source
    tmpdir = None
    if zips:
        tmpdir = Path(tempfile.mkdtemp(prefix="petct_unzip_"))
        print(f"Found {len(zips)} zip archives — extracting to {tmpdir}")
        for z in zips:
            try:
                dest = tmpdir / z.relative_to(args.source).with_suffix("")
                dest.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest)
            except Exception as e:
                print(f"  [warn] could not extract {z.name}: {e}")
        work_root = tmpdir

    try:
        print(f"Scanning {work_root} for DICOM series…")
        series = find_series(work_root)
        if not series:
            raise SystemExit(
                f"No readable DICOM found under {work_root}.\n"
                f"If the data is already NIfTI, pass its directory straight to --data-root."
            )

        groups = group_by_patient(series, work_root)
        by_mod = defaultdict(int)
        for s in series:
            by_mod[s["modality"]] += 1
        print(f"Found {len(series)} series across {len(groups)} patients "
              f"({', '.join(f'{m}:{n}' for m, n in sorted(by_mod.items()))})\n")

        patients = sorted(groups)
        if args.limit:
            patients = patients[:args.limit]
            print(f"--limit {args.limit}: converting the first {len(patients)} patients only\n")

        counts = defaultdict(int)
        for i, pid in enumerate(patients, 1):
            out_dir = args.target / pid
            if (out_dir / "tumorSeg.nii.gz").exists() and not args.overwrite:
                counts["already"] += 1
                continue
            print(f"[{i}/{len(patients)}] {pid}")
            try:
                counts[convert_patient(pid, groups[pid], out_dir)] += 1
            except Exception as e:
                print(f"  [error] {pid}: {e}")
                counts["error"] += 1

        print(f"\nDone. converted={counts['ok']} already-present={counts['already']} "
              f"skipped={counts['skipped']} errors={counts['error']}")
        print(f"\nNext:\n  python scripts/finetune.py --arch base --data-root {args.target} "
              f"--fraction 1.0 --init scratch")
    finally:
        if tmpdir and not args.keep_extracted:
            shutil.rmtree(tmpdir, ignore_errors=True)
        elif tmpdir:
            print(f"\nExtracted DICOM left in {tmpdir}")


if __name__ == "__main__":
    main()
