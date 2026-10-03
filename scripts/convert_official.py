#!/usr/bin/env python3
"""Reconvert the collection with the autoPET organisers' own pipeline.

Our converter mirrored the lesion mask on roughly 63% of tumour-positive scans:
it read the SEG with SimpleITK and then called `CopyInformation(pet)`, stamping
the PET's geometry onto the mask rather than resampling it, which silently
flips the mask wherever the SEG's own orientation differs from the PET's. The
organisers' script (lab-midas/TCIA_processing) handles that case explicitly, and
is also the published standard, so all three outputs come from it rather than
from a flip heuristic applied to our masks.

Their script also differs in two smaller ways that are inherited here on
purpose: the SUV decay is referenced to AcquisitionTime (ours used SeriesTime,
leaving theirs 4-5% higher), and CT is resampled with cubic interpolation and a
fill of -1024 (ours used linear and -1000).

Our study pairing is preserved: scans are fed one study at a time from the
per-study raw layout, so the repeat-visit fix stays in place.

Output names match what the rest of this repo already reads, so no downstream
code changes:

    PET.nii.gz          <- their SUV.nii.gz      (SUV, not Bq/mL)
    CT_resample.nii.gz  <- their CTres.nii.gz    (rounded to int16)
    tumorSeg.nii.gz     <- their SEG.nii.gz

Two deviations, both deliberate:

  * their CTres is float64, which costs ~130 MB per scan; it is rounded to
    int16, the dtype our pipeline already used. The rounding changes a voxel by
    at most 0.5 HU (mean 0.06), far below the 400-4000 HU differences this
    reconversion is fixing.
  * their script calls `pydicom.read_file`, removed in pydicom 3.x and pinned to
    2.3.0 in their requirements. `read_file` was a deprecated alias of
    `dcmread`, so the name is restored rather than editing their code.

    python scripts/convert_official.py --raw-root D:\\data\\dataset \\
        --out-root C:\\data\\autopet_nifti_v2 --jobs 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom

if not hasattr(pydicom, "read_file"):
    pydicom.read_file = pydicom.dcmread

OFFICIAL = r"D:\data\external\TCIA_processing"


def find_scans(raw_root: Path) -> list[tuple[str, Path]]:
    """(case_id, dicom_dir) for every scan, in both flat and per-study layouts."""
    scans = []
    for patient_dir in sorted(d for d in raw_root.iterdir() if d.is_dir()):
        if (patient_dir / "PT.zip").exists():
            scans.append((patient_dir.name, patient_dir))
            continue
        for study_dir in sorted(d for d in patient_dir.iterdir() if d.is_dir()):
            if (study_dir / "PT.zip").exists():
                scans.append((f"{patient_dir.name}_{study_dir.name}", study_dir))
    return scans


def convert_one(job) -> dict:
    case_id, dicom_dir, out_root, tmp_root, overwrite = job
    dicom_dir, out_root = Path(dicom_dir), Path(out_root)

    # mirror our per-scan layout: <patient>/ or <patient>/<study>/
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[2][:2] == "20":
        out_dir = out_root / f"{parts[0]}_{parts[1]}" / "_".join(parts[2:])
    else:
        out_dir = out_root / case_id

    if (out_dir / "tumorSeg.nii.gz").exists() and not overwrite:
        return {"case": case_id, "status": "already"}

    sys.path.insert(0, OFFICIAL)
    import SimpleITK as sitk
    import nibabel as nib
    from tcia_dicom_to_nifti import tcia_to_nifti_study
    import tcia_dicom_to_nifti as tdn

    work = Path(tempfile.mkdtemp(prefix="v2_", dir=tmp_root))
    t0 = time.time()
    try:
        study = work / "patient" / "study"
        for mod in ("CT", "PT", "SEG"):
            zp = dicom_dir / f"{mod}.zip"
            if not zp.exists():
                return {"case": case_id, "status": "missing_zip", "detail": f"no {mod}.zip"}
            dest = study / mod
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                for m in z.namelist():
                    if m.lower().endswith(".dcm"):
                        z.extract(m, dest)

        staged = work / "out"
        staged.mkdir(parents=True, exist_ok=True)
        tdn.nii_out_root = staged
        tcia_to_nifti_study(str(study), str(staged))
        src = staged / "patient" / "study"

        out_dir.mkdir(parents=True, exist_ok=True)
        # SUV becomes the PET channel; it is already float32 from their script
        shutil.copy(src / "SUV.nii.gz", out_dir / "PET.nii.gz")
        shutil.copy(src / "SEG.nii.gz", out_dir / "tumorSeg.nii.gz")

        ct = nib.load(str(src / "CTres.nii.gz"))
        arr = np.asanyarray(ct.dataobj)
        ct16 = nib.Nifti1Image(np.round(arr).astype(np.int16), ct.affine, ct.header)
        ct16.set_data_dtype(np.int16)
        nib.save(ct16, str(out_dir / "CT_resample.nii.gz"))

        seg = np.asanyarray(nib.load(str(out_dir / "tumorSeg.nii.gz")).dataobj)
        return {"case": case_id, "status": "ok", "seconds": round(time.time() - t0, 1),
                "lesion_voxels": int((seg > 0).sum()),
                "shape": [int(v) for v in seg.shape]}
    except Exception:
        return {"case": case_id, "status": "error", "detail": traceback.format_exc()[-300:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--tmp-dir", type=Path, default=Path(r"D:\data\convert_tmp\v2"))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cases", default=None,
                   help="restrict to these case ids: comma-separated, or a file with one per line")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report", type=Path, default=Path(r"D:\data\reports\convert_v2.json"))
    args = p.parse_args()

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    scans = find_scans(args.raw_root)
    if args.cases:
        path = Path(args.cases)
        raw = path.read_text().splitlines() if path.exists() else args.cases.split(",")
        wanted = {c.strip() for c in raw if c.strip()}
        scans = [s for s in scans if s[0] in wanted]
        missing = wanted - {s[0] for s in scans}
        if missing:
            print(f"WARNING: {len(missing)} requested case(s) not found: {sorted(missing)[:5]}")
    if args.limit:
        scans = scans[: args.limit]
    print(f"{len(scans)} scans under {args.raw_root} -> {args.out_root}\n", flush=True)

    jobs = [(c, str(d), str(args.out_root), str(args.tmp_dir), args.overwrite)
            for c, d in scans]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(convert_one, jobs, chunksize=1), 1):
            rows.append(r)
            if r["status"] not in ("ok", "already"):
                print(f"  [{i}] {r['case']}: {r['status']} {r.get('detail','')[:160]}", flush=True)
            if i % 25 == 0:
                done = [x for x in rows if x["status"] == "ok"]
                rate = (time.time() - t0) / max(1, len(done))
                print(f"  {i}/{len(jobs)} | ok {len(done)} | "
                      f"{rate:.1f}s/case wall | ~{(len(jobs)-i)*rate/60:.0f} min left", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] not in ("ok", "already")]
    print(f"\nconverted {len(ok)} | already {sum(1 for r in rows if r['status']=='already')} "
          f"| failed {len(bad)}")
    for r in bad[:15]:
        print(f"   {r['case']}: {r['status']} {r.get('detail','')[:200]}")
    if ok:
        print(f"total lesion voxels: {sum(r['lesion_voxels'] for r in ok):,}")
    args.report.write_text(json.dumps(rows, indent=1))
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
