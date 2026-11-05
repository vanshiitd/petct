#!/usr/bin/env python3
"""Download PSMA-PET-CT-Lesions (autoPET III PSMA cohort) from TCIA and convert
it to NIfTI study by study, deleting the DICOM as soon as each study is done.

The collection is 117 GB of DICOM (378 patients, 597 studies, CC BY 4.0) and
does not fit on this machine's free space, but the converted output does
(~65 MB per study, ~39 GB in all). Each worker therefore:

    download CT/PT/SEG zips (download_tcia.py)  ->  convert with the organisers'
    pipeline (convert_official.py)  ->  verify  ->  delete the zips

so at most `--jobs` studies of DICOM are on disk at once. A free-space guard
stops new studies before the disk fills; rerunning resumes where it stopped.

Output (names match the FDG layout the rest of the repo reads):

    <out>/<PatientID>_<YYYYMMDD>_<uid8>/{PET,CT_resample,tumorSeg}.nii.gz
    <out>/PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv
    <out>/ledger.jsonl                    one line per finished/failed study

Run with the converter venv (SimpleITK, nibabel, pydicom):

    D:\\data\\tciaconv_venv\\Scripts\\python.exe scripts/download_psma.py \\
        --out D:\\data\\psma_nifti --jobs 3
    ... --limit 2          # trial run on the first two patients
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_tcia import (MODALITIES, api_get_json, download_series, group_by_study,
                           install_proxy, normalise_study_date, study_folder_name)
from convert_official import convert_one

COLLECTION = "PSMA-PET-CT-Lesions"
CLINICAL = ("https://www.cancerimagingarchive.net/wp-content/uploads/"
            "PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def study_job(job) -> dict:
    case_id, rows, out_root, tmp_root, retries, min_free, proxy = job
    out_root, tmp_root = Path(out_root), Path(tmp_root)
    install_proxy(proxy)
    if (out_root / case_id / "tumorSeg.nii.gz").exists():
        return {"case": case_id, "status": "already"}
    if free_gb(out_root) < min_free:
        return {"case": case_id, "status": "skipped_low_disk"}
    raw = tmp_root / case_id
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        for mod in MODALITIES:
            dest = raw / f"{mod}.zip"
            if dest.exists():
                continue
            chosen = max((s for s in rows if s.get("Modality") == mod),
                         key=lambda s: int(s.get("ImageCount", 0)))
            if not download_series(chosen["SeriesInstanceUID"], dest, retries, print):
                return {"case": case_id, "status": "download_failed", "modality": mod}
        mb = sum(f.stat().st_size for f in raw.glob("*.zip")) / 1e6
        r = convert_one((case_id, str(raw), case_id, str(out_root), str(tmp_root), False))
        r.update(dicom_mb=round(mb), total_s=round(time.time() - t0, 1))
        return r
    finally:
        # DICOM is never kept: a failed study is simply downloaded again on rerun
        shutil.rmtree(raw, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(r"D:\data\psma_nifti"))
    p.add_argument("--tmp", type=Path, default=Path(r"D:\data\convert_tmp\psma"))
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="first N patients only")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="start no new study below this much free space")
    p.add_argument("--proxy", default=None, metavar="HOST:PORT")
    args = p.parse_args()
    install_proxy(args.proxy)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)

    tsv = args.out / CLINICAL.rsplit("/", 1)[1]
    if not tsv.exists():
        urllib.request.urlretrieve(CLINICAL, tsv)
        print(f"clinical metadata -> {tsv}")

    patients = [r["PatientId"] for r in api_get_json("getPatient", Collection=COLLECTION)]
    if args.limit:
        patients = patients[: args.limit]
    jobs, incomplete = [], 0
    for pid in patients:
        series = api_get_json("getSeries", Collection=COLLECTION, PatientID=pid)
        for uid, rows in group_by_study(series).items():
            if not {s.get("Modality") for s in rows} >= set(MODALITIES):
                incomplete += 1
                continue
            date = next((r.get("StudyDate") for r in rows if r.get("StudyDate")), "")
            jobs.append((f"{pid}_{study_folder_name(uid, date)}", rows, str(args.out),
                         str(args.tmp), args.retries, args.min_free_gb, args.proxy))
    print(f"{len(patients)} patients, {len(jobs)} complete studies "
          f"({incomplete} lacking CT/PT/SEG); free {free_gb(args.out):.0f} GB\n", flush=True)

    counts, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, \
            open(args.out / "ledger.jsonl", "a", encoding="utf-8") as ledger:
        futs = [pool.submit(study_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] != "already":
                ledger.write(json.dumps(r) + "\n")
                ledger.flush()
            if r["status"] not in ("ok", "already"):
                print(f"  {r['case']}: {r['status']} {str(r.get('detail', ''))[-160:]}",
                      flush=True)
            if i % 10 == 0 or i == len(jobs):
                ok = counts.get("ok", 0)
                eta = (time.time() - t0) / max(ok, 1) * (len(jobs) - i) / 60
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {counts} | "
                      f"free {free_gb(args.out):.0f} GB | ~{eta:.0f} min left", flush=True)

    print(f"\nDONE {counts}")
    if any(k not in ("ok", "already") for k in counts):
        print("Rerun the same command to retry; finished studies are skipped.")


if __name__ == "__main__":
    main()
