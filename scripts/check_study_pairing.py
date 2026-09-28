#!/usr/bin/env python3
"""Audit whether each patient's CT, PT and SEG archives come from the SAME study.

TCIA's AutoPET collection has 1,014 studies across 900 patients: 81 patients were
scanned more than once. `download_tcia.py` picks the largest CT, PT and SEG series
*independently*, so for a multi-study patient those three can come from different
visits -- a PET from one date paired with a CT and a lesion mask from another.
`dicom_to_nifti.py` then resamples the mismatched mask onto the PET grid without
complaint, producing a silently wrong training case.

This script reads one DICOM file out of each archive and compares
StudyInstanceUID (0020|000D). It only reads; nothing in the source tree is
modified.

Output: a CSV with one row per patient (patient, per-modality study UID and date,
mismatch flag) and a summary on stdout.

    python scripts/check_study_pairing.py --source D:\\data\\dataset --out audit.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

import SimpleITK as sitk

TAG_STUDY_UID = "0020|000d"
TAG_STUDY_DATE = "0008|0020"
TAG_SERIES_UID = "0020|000e"
MODALITIES = ("CT", "PT", "SEG")


def read_study_tags(dicom_path: Path) -> dict[str, str]:
    """StudyInstanceUID / StudyDate / SeriesInstanceUID from one DICOM file."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(dicom_path))
    r.LoadPrivateTagsOn()
    r.ReadImageInformation()

    def get(tag: str) -> str:
        return r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ""

    return {
        "study_uid": get(TAG_STUDY_UID),
        "study_date": get(TAG_STUDY_DATE),
        "series_uid": get(TAG_SERIES_UID),
    }


def peek_archive(zip_path: Path, tmp_root: Path) -> dict[str, str] | None:
    """Extract the first readable DICOM from an archive and read its study tags.

    Only one member is extracted -- the study UID is identical across a series,
    so there is no reason to unpack hundreds of slices.
    """
    if not zip_path.exists():
        return None
    work = Path(tempfile.mkdtemp(prefix="pairing_", dir=tmp_root))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            for member in sorted(members):
                target = work / Path(member).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                try:
                    return read_study_tags(target)
                except Exception:
                    target.unlink(missing_ok=True)
                    continue  # not a readable DICOM; try the next member
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def find_scan_dirs(patient_dir: Path) -> list[Path]:
    """Directories holding one scan's archives.

    Two layouts exist: a single-study patient keeps its archives directly in the
    patient folder, while a multi-study patient has one sub-directory per study.
    """
    if any((patient_dir / f"{m}.zip").exists() for m in MODALITIES):
        return [patient_dir]
    return sorted(d for d in patient_dir.iterdir()
                  if d.is_dir() and any((d / f"{m}.zip").exists() for m in MODALITIES))


def audit_patient(patient_dir: Path, tmp_root: Path, scan_dir: Path | None = None) -> dict:
    scan_dir = scan_dir or patient_dir
    row: dict[str, str | bool] = {
        "patient": patient_dir.name,
        "scan": "" if scan_dir == patient_dir else scan_dir.name,
    }
    uids: list[str] = []
    for mod in MODALITIES:
        tags = peek_archive(scan_dir / f"{mod}.zip", tmp_root)
        if tags is None:
            row[f"{mod}_study_uid"] = ""
            row[f"{mod}_study_date"] = ""
            continue
        row[f"{mod}_study_uid"] = tags["study_uid"]
        row[f"{mod}_study_date"] = tags["study_date"]
        if tags["study_uid"]:
            uids.append(tags["study_uid"])

    present = [m for m in MODALITIES if row.get(f"{m}_study_uid")]
    row["n_modalities"] = len(present)
    row["missing"] = ",".join(m for m in MODALITIES if m not in present)
    # A mismatch needs all three present; an incomplete patient is reported
    # separately rather than being called a mismatch.
    row["mismatch"] = bool(len(present) == len(MODALITIES) and len(set(uids)) > 1)
    row["n_distinct_studies"] = len(set(uids))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True,
                   help="root of the raw download (one folder per patient)")
    p.add_argument("--out", type=Path, required=True, help="CSV to write")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="scratch directory for the single-file extractions")
    p.add_argument("--limit", type=int, default=None, help="audit at most N patients")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    tmp_root = args.tmp_dir or Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)

    patients = sorted(d for d in args.source.iterdir() if d.is_dir())
    if args.limit:
        patients = patients[: args.limit]
    print(f"Auditing {len(patients)} patient folders under {args.source}\n")

    rows = []
    for i, patient_dir in enumerate(patients, 1):
        scan_dirs = find_scan_dirs(patient_dir)
        if not scan_dirs:
            print(f"[{i}/{len(patients)}] {patient_dir.name}: no archives found")
            continue
        for scan_dir in scan_dirs:
            row = audit_patient(patient_dir, tmp_root, scan_dir)
            rows.append(row)
            flag = "MISMATCH" if row["mismatch"] else ("incomplete" if row["missing"] else "ok")
            if flag != "ok" or i % 100 == 0:
                name = row["patient"] + (f"/{row['scan']}" if row["scan"] else "")
                print(f"[{i}/{len(patients)}] {name}: {flag}")

    fields = ["patient", "scan", "CT_study_uid", "CT_study_date", "PT_study_uid", "PT_study_date",
              "SEG_study_uid", "SEG_study_date", "n_modalities", "missing",
              "n_distinct_studies", "mismatch"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    def label(r: dict) -> str:
        return r["patient"] + (f"/{r['scan']}" if r["scan"] else "")

    mismatched = [label(r) for r in rows if r["mismatch"]]
    incomplete = [label(r) for r in rows if r["missing"]]
    print(f"\n=== Summary ===")
    print(f"patients audited : {len({r['patient'] for r in rows})}")
    print(f"scans audited    : {len(rows)}")
    print(f"complete (3 mods): {sum(1 for r in rows if r['n_modalities'] == 3)}")
    print(f"incomplete       : {len(incomplete)} {incomplete if incomplete else ''}")
    print(f"MISMATCHED       : {len(mismatched)}")
    for pid in mismatched:
        print(f"   {pid}")
    print(f"\nCSV written to {args.out}")


if __name__ == "__main__":
    main()
