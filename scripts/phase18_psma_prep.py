#!/usr/bin/env python3
"""Phase XVIII: prepare PSMA for the unchanged FDG pipeline.

See `reports/phase18_psma_plan.md`. For each patient's earliest converted
study:
  1. resample PET (linear), CT (linear, fill -1024, int16) and mask (nearest)
     onto the FDG grid, 2.0364 x 2.0364 x 3.0 mm, keeping origin and direction;
  2. export it through `export_nnunet_maeprep.export_case` into Dataset505
     imagesTs / labelsTs_prep / inverse (new PSMA_* files only).

Writes reports/cases_psma.txt and D:\\data\\reports\\phase18_psma_prep.json.
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_nnunet_maeprep import export_case

SRC = Path(r"D:\data\psma_nifti")
DST = Path(r"D:\data\psma_fdggrid")
DS = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
SPACING = (2.0364201068878174, 2.0364201068878174, 3.0)   # FDG grid, x y z


def earliest_per_patient() -> list[str]:
    ok = [json.loads(l) for l in open(SRC / "ledger.jsonl", encoding="utf-8")]
    ok = {r["case"] for r in ok if r["status"] == "ok"}
    by_p = {}
    for c in sorted(ok):                       # PSMA_<pid>_<YYYYMMDD>_<uid8>
        pid, date, uid = c.rsplit("_", 2)
        by_p.setdefault(pid, []).append((date, uid, c))
    return sorted(min(v)[2] for v in by_p.values())


def resample(img, interp, fill, pixel=None):
    size = [int(math.ceil(n * s / t)) for n, s, t in
            zip(img.GetSize(), img.GetSpacing(), SPACING)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), SPACING,
                         img.GetDirection(), fill, pixel or img.GetPixelID())


def prep(case: str) -> dict:
    out = DST / case
    if not (out / "tumorSeg.nii.gz").exists():
        out.mkdir(parents=True, exist_ok=True)
        pet = sitk.ReadImage(str(SRC / case / "PET.nii.gz"))
        ref = resample(pet, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        ct = sitk.ReadImage(str(SRC / case / "CT_resample.nii.gz"), sitk.sitkFloat32)
        ct = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
        seg = sitk.ReadImage(str(SRC / case / "tumorSeg.nii.gz"))
        seg = sitk.Resample(seg, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(ref, str(out / "PET.nii.gz"))
        sitk.WriteImage(sitk.Cast(sitk.Round(ct), sitk.sitkInt16), str(out / "CT_resample.nii.gz"))
        sitk.WriteImage(seg, str(out / "tumorSeg.nii.gz"))
    if (DS / "inverse" / f"{case}.json").exists():
        return {"case": case, "status": "already"}
    return export_case((case, str(out), str(DS), "Ts", "labelsTs_prep", None, None, None))


def main() -> None:
    cases = earliest_per_patient()
    print(f"{len(cases)} patients -> earliest study each", flush=True)
    Path(__file__).resolve().parent.parent.joinpath("reports", "cases_psma.txt") \
        .write_text("\n".join(cases) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, r in enumerate(pool.map(prep, cases, chunksize=2), 1):
            rows.append(r)
            if r["status"] not in ("ok", "already"):
                print(f"  {r['case']}: {r['status']} {r.get('detail', '')}", flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}", flush=True)
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(f"done {st}")
    Path(r"D:\data\reports\phase18_psma_prep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
