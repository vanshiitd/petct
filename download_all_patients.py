"""
Downloads CT + PET + SEG for all patients in TCIA's FDG-PET-CT-Lesions
collection (~419GB total, ~900 patients). Safe to stop (Ctrl+C) and rerun --
it skips any patient already fully downloaded.

No dependencies beyond the Python standard library -- works anywhere Python
runs, including machines where running .sh/.ps1 scripts is restricted.

Usage:
    python download_all_patients.py [output_dir]

Defaults to ./AutoPET_Full if no output_dir is given.
"""
import json
import socket
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

socket.setdefaulttimeout(300)

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "FDG-PET-CT-Lesions"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./AutoPET_Full")
LOG_FILE = OUT_DIR / "download_log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def api_get_json(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{path}?{qs}" if qs else f"{API}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def download_series(uid, dest_zip):
    url = f"{API}/getImage?SeriesInstanceUID={uid}"
    urllib.request.urlretrieve(url, dest_zip)


def process_patient(patient_id, series):
    patient_dir = OUT_DIR / patient_id
    done_marker = patient_dir / ".done"
    patient_dir.mkdir(parents=True, exist_ok=True)

    ct_series = max(
        (s for s in series if s["Modality"] == "CT"),
        key=lambda s: int(s.get("ImageCount", 0)),
        default=None,
    )
    pt_series = next((s for s in series if s["Modality"] == "PT"), None)
    seg_series = next((s for s in series if s["Modality"] == "SEG"), None)

    targets = {"CT": ct_series, "PT": pt_series, "SEG": seg_series}
    all_ok = True

    for tag, s in targets.items():
        if s is None:
            log(f"  SKIP {patient_id}: missing {tag} series")
            all_ok = False
            continue

        dest_zip = patient_dir / f"{tag}.zip"
        if dest_zip.exists():
            continue  # already downloaded this one

        tmp_zip = patient_dir / f"{tag}.zip.part"
        size_mb = s.get("FileSize", 0) / 1e6
        try:
            log(f"  downloading {tag} (~{size_mb:.0f}MB)...")
            download_series(s["SeriesInstanceUID"], tmp_zip)
            # sanity-check the zip isn't truncated/corrupt before keeping it
            with zipfile.ZipFile(tmp_zip) as zf:
                bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt entry in zip: {bad}")
            tmp_zip.rename(dest_zip)
        except Exception as e:
            log(f"  ERROR downloading {tag} for {patient_id}: {e}")
            tmp_zip.unlink(missing_ok=True)
            all_ok = False

    if all_ok:
        done_marker.touch()
        log(f"  OK {patient_id}")

    return all_ok


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Fetching patient list for {COLLECTION}...")
    patients = api_get_json("getPatient", Collection=COLLECTION)
    total = len(patients)
    log(f"Found {total} patients.")

    for i, p in enumerate(patients, 1):
        patient_id = p["PatientId"]
        done_marker = OUT_DIR / patient_id / ".done"

        if done_marker.exists():
            log(f"[{i}/{total}] {patient_id}: already done, skipping")
            continue

        log(f"[{i}/{total}] {patient_id}: fetching series list...")
        try:
            series = api_get_json("getSeries", Collection=COLLECTION, PatientID=patient_id)
        except Exception as e:
            log(f"  ERROR fetching series list for {patient_id}: {e}")
            continue

        process_patient(patient_id, series)

    log(f"DONE. Check {LOG_FILE} for any SKIP/ERROR lines to retry individually.")


if __name__ == "__main__":
    main()
