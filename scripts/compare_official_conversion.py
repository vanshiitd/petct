"""Run the organisers' DICOM->NIfTI conversion and diff it against ours.

Their script (lab-midas/TCIA_processing) differs from ours in ways that could
matter, so this runs it unmodified on a sample of patients and compares voxel by
voxel: their SUV against our SUV.nii.gz, their CTres against our
CT_resample.nii.gz, their SEG against our tumorSeg.nii.gz.

They convert with dicom2nifti(reorient=True), which may put the arrays in a
different axis order from ours, so everything is compared through physical
space: theirs is resampled onto our grid with SimpleITK before differencing
(nearest neighbour for the mask, linear for images). Geometry is reported
separately so a pure orientation difference is visible rather than hidden.
"""
import json, pathlib, shutil, sys, tempfile, traceback, zipfile
import numpy as np
import pydicom

# Their script calls pydicom.read_file, which pydicom 3.x removed; their
# requirements pin pydicom 2.3.0. read_file was a deprecated alias of dcmread,
# so restoring the name runs their code unchanged rather than downgrading the
# whole stack or editing their file.
if not hasattr(pydicom, "read_file"):
    pydicom.read_file = pydicom.dcmread

sys.path.insert(0, r"D:\data\external\TCIA_processing")

RAW = pathlib.Path(r"D:\data\dataset")
OURS = pathlib.Path(r"D:\data\autopet_nifti")
WORK = pathlib.Path(r"D:\data\official_conv")
TMP = pathlib.Path(r"D:\data\convert_tmp\offconv")


def our_dirs(case: str):
    d = OURS / case
    if (d / "PET.nii.gz").exists():
        return d, RAW / case
    parts = case.split("_")
    patient = f"{parts[0]}_{parts[1]}"
    rel = case[len(patient) + 1:]
    return OURS / patient / rel, RAW / patient / rel


def stage_dicom(raw_dir: pathlib.Path, dest: pathlib.Path) -> bool:
    """Unpack CT/PT/SEG zips into the layout their script expects."""
    for mod in ("CT", "PT", "SEG"):
        zp = raw_dir / f"{mod}.zip"
        if not zp.exists():
            return False
        out = dest / mod
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.lower().endswith(".dcm"):
                    z.extract(m, out)
    return True


def compare(case: str) -> dict:
    import SimpleITK as sitk
    from tcia_dicom_to_nifti import tcia_to_nifti_study

    ours_dir, raw_dir = our_dirs(case)
    work = pathlib.Path(tempfile.mkdtemp(prefix="oc_", dir=TMP))
    try:
        study = work / "patient" / "study"
        if not stage_dicom(raw_dir, study):
            return {"case": case, "status": "missing_zip"}

        out_root = work / "out"
        out_root.mkdir(parents=True, exist_ok=True)
        import tcia_dicom_to_nifti as tdn
        tdn.nii_out_root = out_root                 # the script reads this global
        tcia_to_nifti_study(str(study), str(out_root))
        theirs = out_root / "patient" / "study"

        res = {"case": case, "status": "ok"}
        pairs = [("SUV", theirs / "SUV.nii.gz", ours_dir / "SUV.nii.gz", False),
                 ("CTres", theirs / "CTres.nii.gz", ours_dir / "CT_resample.nii.gz", False),
                 ("SEG", theirs / "SEG.nii.gz", ours_dir / "tumorSeg.nii.gz", True)]
        for name, tpath, opath, is_mask in pairs:
            if not tpath.exists() or not opath.exists():
                res[name] = {"status": "missing"}
                continue
            t = sitk.ReadImage(str(tpath))
            o = sitk.ReadImage(str(opath))
            geom = {
                "their_size": list(t.GetSize()), "our_size": list(o.GetSize()),
                "their_spacing": [round(v, 5) for v in t.GetSpacing()],
                "our_spacing": [round(v, 5) for v in o.GetSpacing()],
                "their_origin": [round(v, 3) for v in t.GetOrigin()],
                "our_origin": [round(v, 3) for v in o.GetOrigin()],
                "same_direction": bool(np.allclose(t.GetDirection(), o.GetDirection(), atol=1e-6)),
                "same_grid": bool(t.GetSize() == o.GetSize()
                                  and np.allclose(t.GetSpacing(), o.GetSpacing(), atol=1e-4)
                                  and np.allclose(t.GetOrigin(), o.GetOrigin(), atol=1e-3)
                                  and np.allclose(t.GetDirection(), o.GetDirection(), atol=1e-6)),
            }
            # bring theirs onto our grid through physical space
            interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
            t_on_ours = t if geom["same_grid"] else sitk.Resample(
                t, o, sitk.Transform(), interp, -1024.0 if name == "CTres" else 0.0,
                t.GetPixelID())
            ta = sitk.GetArrayFromImage(t_on_ours).astype(np.float64)
            oa = sitk.GetArrayFromImage(o).astype(np.float64)
            if is_mask:
                tb, ob = ta > 0, oa > 0
                denom = int(tb.sum()) + int(ob.sum())
                geom["dice"] = 1.0 if denom == 0 else 2.0 * int((tb & ob).sum()) / denom
                geom["their_voxels"] = int(tb.sum())
                geom["our_voxels"] = int(ob.sum())
            else:
                diff = np.abs(ta - oa)
                geom["max_abs_diff"] = float(diff.max())
                geom["mean_abs_diff"] = float(diff.mean())
                geom["their_mean"] = float(ta.mean())
                geom["our_mean"] = float(oa.mean())
                scale = ta[oa != 0] / oa[oa != 0] if (oa != 0).any() else np.array([np.nan])
                geom["median_ratio_theirs_over_ours"] = float(np.median(scale))
            res[name] = geom
        return res
    except Exception:
        return {"case": case, "status": "error", "detail": traceback.format_exc()[-400:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    cases = [c.strip() for c in pathlib.Path(sys.argv[1]).read_text().splitlines() if c.strip()]
    TMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(cases, 1):
        r = compare(c)
        rows.append(r)
        if r["status"] != "ok":
            print(f"[{i}/{len(cases)}] {c}: {r['status']} {r.get('detail','')[:200]}", flush=True)
            continue
        seg = r.get("SEG", {})
        suv = r.get("SUV", {})
        ct = r.get("CTres", {})
        print(f"[{i}/{len(cases)}] {c[:32]:32s} "
              f"SEG dice {seg.get('dice', float('nan')):.4f} "
              f"| SUV ratio {suv.get('median_ratio_theirs_over_ours', float('nan')):.4f} "
              f"maxdiff {suv.get('max_abs_diff', float('nan')):.3f} "
              f"| CT maxdiff {ct.get('max_abs_diff', float('nan')):.1f} "
              f"| same grid SUV/CT/SEG {suv.get('same_grid')}/{ct.get('same_grid')}/{seg.get('same_grid')}",
              flush=True)
    pathlib.Path(r"D:\data\reports\official_conversion_diff.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwritten to D:\\data\\reports\\official_conversion_diff.json")


if __name__ == "__main__":
    main()
