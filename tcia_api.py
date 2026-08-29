"""
Minimal client for TCIA's public NBIA REST API — no auth needed for
open-access collections like FDG-PET-CT-Lesions (the AutoPET data).

Usage:
    from tcia_api import list_patients, list_series, download_series

    patients = list_patients()                       # 900 patients
    series = list_series(patients[0]["PatientId"])    # that patient's CT/PT/SEG
    ct = next(s for s in series if s["Modality"] == "CT")
    download_series(ct["SeriesInstanceUID"], "ct.zip")
"""
import json
import urllib.request

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "FDG-PET-CT-Lesions"


def _get(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{path}?{qs}" if qs else f"{API}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def list_collections():
    """All public TCIA collections."""
    return _get("getCollectionValues")


def list_patients(collection=COLLECTION):
    """All patients in a collection. Each has a 'PatientId'."""
    return _get("getPatient", Collection=collection)


def list_series(patient_id, collection=COLLECTION):
    """A patient's series (CT, PT, SEG). Each has 'SeriesInstanceUID',
    'Modality', 'ImageCount', 'FileSize'."""
    return _get("getSeries", Collection=collection, PatientID=patient_id)


def download_series(series_uid, dest_zip):
    """Download one series (a full CT, PET, or SEG) as a ZIP of DICOM files."""
    url = f"{API}/getImage?SeriesInstanceUID={series_uid}"
    urllib.request.urlretrieve(url, dest_zip)


if __name__ == "__main__":
    patients = list_patients()
    print(f"{len(patients)} patients in {COLLECTION}")
    print("first patient:", patients[0]["PatientId"])
    series = list_series(patients[0]["PatientId"])
    for s in series:
        print(f"  {s['Modality']:4s}  {s['ImageCount']:>5} images  "
              f"{s['FileSize']/1e6:6.1f} MB  {s['SeriesInstanceUID']}")
