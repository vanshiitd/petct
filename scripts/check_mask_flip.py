"""How many of our masks are misplaced?

The PET decides: a correctly placed lesion label sits on elevated FDG uptake. A
mask that is flipped in-plane lands on whatever happens to be mirrored across
the body, which is usually unremarkable tissue. So for every tumour-positive
scan this compares the PET inside our mask against the PET inside the mask
flipped left-right, and asks which is brighter.

Also reads each SEG's ImageOrientationPatient, since the organisers' script
keys its flip on element 4 of that tag.
"""
import json, pathlib, shutil, tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pydicom
import SimpleITK as sitk

OURS = pathlib.Path(r"D:\data\autopet_nifti")
RAW = pathlib.Path(r"D:\data\dataset")
TMP = pathlib.Path(r"D:\data\convert_tmp\flipscope")


def dirs_for(case):
    d = OURS / case
    if (d / "PET.nii.gz").exists():
        return d, RAW / case
    parts = case.split("_")
    p = f"{parts[0]}_{parts[1]}"
    return OURS / p / case[len(p) + 1:], RAW / p / case[len(p) + 1:]


def one(case):
    ours_dir, raw_dir = dirs_for(case)
    try:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ours_dir / "tumorSeg.nii.gz"))) > 0
        if not seg.any():
            return None
        pet = sitk.GetArrayFromImage(sitk.ReadImage(str(ours_dir / "PET.nii.gz")))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ours_dir / "CT_resample.nii.gz")))
        body = ct > -500
        bg = float(np.median(pet[body & ~seg])) if (body & ~seg).any() else 1.0
        bg = max(bg, 1.0)

        as_is = float(np.median(pet[seg])) / bg
        # the flip the organisers apply: left-right within each axial slice.
        # arrays are (z, y, x); their np.flip(mask,1) acts on the second axis of
        # a (x, y, z) array, i.e. y -- which is axis 1 here too.
        flipped = seg[:, ::-1, :]
        as_flipped = float(np.median(pet[flipped])) / bg

        # orientation tag the organisers key on
        orient4 = None
        work = pathlib.Path(tempfile.mkdtemp(prefix="fs_", dir=TMP))
        try:
            with zipfile.ZipFile(raw_dir / "SEG.zip") as z:
                m = sorted(n for n in z.namelist() if n.lower().endswith(".dcm"))
                if m:
                    z.extract(m[0], work)
                    ds = pydicom.dcmread(str(work / m[0]), stop_before_pixels=True)
                    sh = ds.get("SharedFunctionalGroupsSequence", [None])[0]
                    if sh is not None and "PlaneOrientationSequence" in sh:
                        iop = sh.PlaneOrientationSequence[0].ImageOrientationPatient
                        orient4 = float(iop[4])
        finally:
            shutil.rmtree(work, ignore_errors=True)

        return {"case": case, "ratio_as_is": as_is, "ratio_flipped": as_flipped,
                "flip_better": bool(as_flipped > as_is * 1.2),
                "orientation_4": orient4, "voxels": int(seg.sum())}
    except Exception as e:
        return {"case": case, "error": repr(e)[:150]}


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    split = json.load(open(r"D:\data\petct\splits\autopet_v1.json"))
    flags = json.load(open(r"D:\data\reports\tumour_flags.json"))
    cases = [c for s in ("train", "val", "test")
             for cs in split["splits"][s]["cases"].values() for c in cs]
    pos = [c for c in cases if flags.get(c)]
    print(f"checking {len(pos)} tumour-positive scans")

    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, r in enumerate(pool.map(one, pos, chunksize=4), 1):
            if r:
                rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(pos)}", flush=True)

    ok = [r for r in rows if "error" not in r]
    flip_better = [r for r in ok if r["flip_better"]]
    low = [r for r in ok if r["ratio_as_is"] < 1.5]
    print(f"\nscans measured: {len(ok)}")
    print(f"mask sits on BRIGHTER tissue when flipped : {len(flip_better)} "
          f"({len(flip_better)/max(1,len(ok)):.1%})")
    print(f"mask uptake ratio below 1.5x as stored    : {len(low)} "
          f"({len(low)/max(1,len(ok)):.1%})")
    o4 = {}
    for r in ok:
        o4[r["orientation_4"]] = o4.get(r["orientation_4"], 0) + 1
    print(f"ImageOrientationPatient[4] values: {o4}")
    byo = {}
    for r in ok:
        k = (r["orientation_4"], r["flip_better"])
        byo[k] = byo.get(k, 0) + 1
    print(f"(orientation[4], flip_is_better) counts: {byo}")
    r_as = np.array([r["ratio_as_is"] for r in ok])
    r_fl = np.array([r["ratio_flipped"] for r in ok])
    print(f"\nuptake ratio as stored : median {np.median(r_as):.2f} | "
          f"p5 {np.percentile(r_as,5):.2f} | p95 {np.percentile(r_as,95):.2f}")
    print(f"uptake ratio flipped   : median {np.median(r_fl):.2f} | "
          f"p5 {np.percentile(r_fl,5):.2f} | p95 {np.percentile(r_fl,95):.2f}")
    json.dump(rows, open(r"D:\data\reports\mask_flip_scope.json", "w"), indent=1)
    print("\nwritten to D:\\data\\reports\\mask_flip_scope.json")
