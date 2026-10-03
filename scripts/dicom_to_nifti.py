#!/usr/bin/env python3
"""Convert a downloaded TCIA PET/CT collection into the NIfTI layout the
training pipeline expects.

TCIA ships raw DICOM. Every other entry point in this repo expects, per scan:

    <out>/<PatientID>/PET.nii.gz
    <out>/<PatientID>/CT_resample.nii.gz
    <out>/<PatientID>/tumorSeg.nii.gz

A patient scanned more than once gets one directory per study,
`<out>/<PatientID>/<YYYYMMDD>_<uid8>/…`, which `build_subject_index` already
understands (it takes the patient ID from the first path component and globs
for PET.nii.gz at any depth).

CT, PT and SEG must come from the SAME study: a mask resampled from another
visit is silently wrong, and a CT from another visit can land outside the PET's
field of view and convert to a uniformly -1000 HU volume. Such a case is
reported and skipped, never written.

This script bridges that gap. It auto-detects what it is given:

  * directories of .dcm files  (e.g. an NBIA Data Retriever download)
  * per-patient .zip archives  (one CT/PT/SEG archive per patient folder)
  * already-converted .nii.gz  (reported and skipped)

Series are identified by the DICOM Modality tag, not by folder name, so it does
not care how the download tool arranged things. CT is resampled onto the PET
grid so the two are voxel-aligned, and the segmentation is binarised.

Zip archives are unpacked one patient at a time and deleted straight after,
so scratch use stays at ~2 GB however large the collection is (use --tmp-dir to
choose the drive). Safe to interrupt and rerun: converted patients are skipped.

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
TAG_STUDY_UID = "0020|000d"
TAG_STUDY_DATE = "0008|0020"


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
                "study_uid": (r.GetMetaData(TAG_STUDY_UID).strip()
                              if r.HasMetaDataKey(TAG_STUDY_UID) else None),
                "study_date": (r.GetMetaData(TAG_STUDY_DATE).strip()
                               if r.HasMetaDataKey(TAG_STUDY_DATE) else ""),
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


def split_by_study(series: list[dict]) -> dict[str, list[dict]]:
    """Group a patient's series by StudyInstanceUID (one entry per visit)."""
    studies: dict[str, list[dict]] = defaultdict(list)
    for s in series:
        studies[s.get("study_uid") or "unknown"].append(s)
    return dict(studies)


def study_label(series: list[dict]) -> str:
    """'<YYYYMMDD>_<uid8>' -- matches the downloader's directory naming."""
    uid = next((s.get("study_uid") for s in series if s.get("study_uid")), "")
    date = next((s.get("study_date") for s in series if s.get("study_date")), "")
    return f"{date or 'nodate'}_{uid[-8:] if uid else 'nouid'}"


def convert_patient(patient_id: str, series: list[dict], out_dir: Path) -> str:
    """Write PET / CT_resample / tumorSeg for one scan. Returns a status word."""
    cts = [s for s in series if s["modality"] == "CT"]
    pts = [s for s in series if s["modality"] in ("PT", "PET")]
    segs = [s for s in series if s["modality"] == "SEG"]

    missing = [n for n, v in (("CT", cts), ("PT", pts), ("SEG", segs)) if not v]
    if missing:
        print(f"  [skip] {patient_id}: missing {', '.join(missing)}")
        return "skipped"

    # Refuse to build a scan out of different visits. Resampling a mask from
    # one study onto another study's PET grid produces a plausible-looking but
    # silently wrong training case -- and a CT from the wrong visit can land
    # entirely outside the PET's field of view, giving a uniformly -1000 HU
    # volume. Report it and move on rather than writing something wrong.
    study_uids = {s.get("study_uid") for s in (cts[:1] + pts[:1] + segs[:1])}
    if len(study_uids) > 1:
        dates = {m: next((s.get("study_date") for s in v), "?")
                 for m, v in (("CT", cts), ("PT", pts), ("SEG", segs))}
        print(f"  [MISMATCH] {patient_id}: CT/PT/SEG come from different studies "
              f"(CT {dates['CT']}, PT {dates['PT']}, SEG {dates['SEG']}) -- refusing")
        return "mismatch"

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

    # Always resample the mask onto the PET grid through physical space, with
    # nearest neighbour so no label value is invented by interpolation.
    #
    # This used to call CopyInformation(pet) whenever the two had the same
    # size, which stamps the PET's geometry onto the mask instead of moving the
    # mask onto the PET's grid. Matching sizes do not imply matching geometry:
    # for 317 of this collection's 501 tumour-positive scans the SEG's
    # orientation differs from the PET's, and the shortcut silently mirrored the
    # lesion left-right. Models trained on those labels could not fit them --
    # training Dice equalled test Dice at 0.29 -- and nothing downstream
    # revealed it, because the mask stayed the right shape and the right size.
    #
    # Resampling is correct whether or not the geometries agree, and costs
    # nothing when they do.
    seg_bin = sitk.Resample(sitk.Cast(seg > 0, sitk.sitkUInt8), pet, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

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
def convert_zip_layout(zips: list[Path], args) -> dict[str, int]:
    """Convert per-patient zip archives, unpacking ONE patient at a time.

    Unpacking the whole collection first would need the full uncompressed
    DICOM size (~400 GB for AutoPET) free in the temp directory at once.
    Unpacking per patient and deleting straight after keeps peak scratch use
    to a single patient (~1-2 GB).
    """
    # Key by the archive's directory, so a multi-study patient's
    # <PatientID>/<study>/ folders are each converted as their own scan while a
    # single-study patient's <PatientID>/ folder still works.
    by_folder: dict[str, list[Path]] = defaultdict(list)
    for z in zips:
        parts = z.relative_to(args.source).parts
        by_folder["/".join(parts[:-1]) if len(parts) > 1 else z.stem].append(z)

    folders = sorted(by_folder)
    if args.limit:
        folders = folders[:args.limit]
        print(f"--limit {args.limit}: converting the first {len(folders)} patients only\n")

    counts: dict[str, int] = defaultdict(int)
    for i, folder in enumerate(folders, 1):
        if (args.target / folder / "tumorSeg.nii.gz").exists() and not args.overwrite:
            counts["already"] += 1
            continue

        print(f"[{i}/{len(folders)}] {folder}")
        work = Path(tempfile.mkdtemp(prefix="petct_unzip_", dir=args.tmp_dir))
        try:
            # keep the patient folder name as the top-level directory, so the
            # PatientID fallback in group_by_patient still resolves to it
            for z in by_folder[folder]:
                dest = work / folder / z.stem
                dest.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest)

            groups = group_by_patient(find_series(work), work)
            if not groups:
                print(f"  [skip] {folder}: no readable DICOM in its archives")
                counts["skipped"] += 1
                continue
            # one archive folder is one scan: <PatientID>/ for a single-study
            # patient, <PatientID>/<study>/ for a multi-study one. Mirror that
            # path into the output tree.
            for pid, series in groups.items():
                counts[convert_patient(folder, series, args.target / folder)] += 1
        except Exception as e:
            print(f"  [error] {folder}: {e}")
            counts["error"] += 1
        finally:
            if args.keep_extracted:
                print(f"  extracted DICOM left in {work}")
            else:
                shutil.rmtree(work, ignore_errors=True)
    return counts


def convert_dicom_dirs(root: Path, args) -> dict[str, int]:
    """Convert a tree of already-unpacked DICOM directories."""
    print(f"Scanning {root} for DICOM series…")
    series = find_series(root)
    if not series:
        raise SystemExit(
            f"No readable DICOM found under {root}.\n"
            f"If the data is already NIfTI, pass its directory straight to --data-root."
        )

    groups = group_by_patient(series, root)
    by_mod: dict[str, int] = defaultdict(int)
    for s in series:
        by_mod[s["modality"]] += 1
    print(f"Found {len(series)} series across {len(groups)} patients "
          f"({', '.join(f'{m}:{n}' for m, n in sorted(by_mod.items()))})\n")

    patients = sorted(groups)
    if args.limit:
        patients = patients[:args.limit]
        print(f"--limit {args.limit}: converting the first {len(patients)} patients only\n")

    counts: dict[str, int] = defaultdict(int)
    for i, pid in enumerate(patients, 1):
        # An unpacked DICOM tree can hold several visits for one patient; each
        # becomes its own scan rather than being merged into one (which would
        # pair images and mask from different dates).
        studies = split_by_study(groups[pid])
        multi = len(studies) > 1
        print(f"[{i}/{len(patients)}] {pid}" + (f" ({len(studies)} studies)" if multi else ""))
        for series in studies.values():
            out_dir = args.target / pid / study_label(series) if multi else args.target / pid
            label = f"{pid}/{study_label(series)}" if multi else pid
            if (out_dir / "tumorSeg.nii.gz").exists() and not args.overwrite:
                counts["already"] += 1
                continue
            try:
                counts[convert_patient(label, series, out_dir)] += 1
            except Exception as e:
                print(f"  [error] {label}: {e}")
                counts["error"] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="root of the downloaded collection")
    p.add_argument("--target", type=Path, required=True, help="output root for the NIfTI layout")
    p.add_argument("--limit", type=int, default=None, help="convert at most N patients (for a trial run)")
    p.add_argument("--overwrite", action="store_true", help="reconvert patients already present")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="where to unpack zip archives (default: the system temp dir). "
                        "Needs ~2 GB free per patient; point it at a large drive if C: is small.")
    p.add_argument("--keep-extracted", action="store_true",
                   help="do not delete the unpacked DICOM after each patient (debugging; uses a lot of disk)")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # already-converted?
    existing_nii = list(args.source.rglob("PET.nii.gz"))
    if existing_nii:
        print(f"Found {len(existing_nii)} PET.nii.gz already under {args.source}.")
        print("This data appears to be converted already — point --data-root at it directly:")
        print(f"  python scripts/finetune.py --arch base --data-root {args.source} ...")
        if not list(args.source.rglob("*.dcm")) and not list(args.source.rglob("*.zip")):
            return

    zips = sorted(args.source.rglob("*.zip"))
    if zips:
        print(f"Found {len(zips)} zip archives — unpacking one patient at a time.\n")
        counts = convert_zip_layout(zips, args)
    else:
        counts = convert_dicom_dirs(args.source, args)

    print(f"\nDone. converted={counts['ok']} already-present={counts['already']} "
          f"skipped={counts['skipped']} mismatched={counts['mismatch']} "
          f"errors={counts['error']}")
    if counts["mismatch"]:
        print(f"{counts['mismatch']} scan(s) refused because CT/PT/SEG came from different "
              f"studies. Re-download those patients with the current downloader, which "
              f"pairs within a study:\n"
              f"  python scripts/download_tcia.py <raw_dir> --patients <ids>")
    if counts["error"] or counts["skipped"]:
        print("Rerun the same command to retry; finished patients are skipped.")
    print(f"\nNext:\n  python scripts/finetune.py --arch small --data-root {args.target} "
          f"--fraction 1.0 --init foundation")


if __name__ == "__main__":
    main()
