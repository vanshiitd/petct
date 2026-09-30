#!/usr/bin/env python3
"""Write SUV.nii.gz next to each PET.nii.gz, without re-converting from DICOM.

SUVbw is the activity concentration times a single scalar per scan:

    SUVbw = PET[Bq/mL] * patient_weight[g] / decayed_dose[Bq]

so the volume itself never has to be rebuilt. This reads one DICOM header per
scan for the factor, multiplies the existing PET.nii.gz by it, and writes
SUV.nii.gz alongside. PET.nii.gz is left exactly as it is.

The factor and every tag it came from go to a CSV, and scans with missing or
implausible tags are listed rather than guessed at -- a wrong dose or weight
produces a plausible-looking volume with silently wrong numbers, which is worse
than a missing file.

Output is float32: the source PET is float64, which costs twice the disk for
precision far beyond what a scanner delivers.

    python scripts/make_suv.py --data-root D:\\data\\autopet_nifti \\
        --raw-root D:\\data\\dataset --csv D:\\data\\reports\\suv_factors.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from petct.suv import suv_factor_from_file  # noqa: E402


def raw_archive_for(scan_dir: Path, data_root: Path, raw_root: Path) -> Path | None:
    """The PT.zip that produced this scan.

    Mirrors the layout: <pid>/ for a single-study patient, <pid>/<study>/ for a
    multi-study one, identical on both sides.
    """
    rel = scan_dir.relative_to(data_root)
    candidate = raw_root / rel / "PT.zip"
    return candidate if candidate.exists() else None


def factor_for_archive(zip_path: Path, tmp_dir: Path):
    """Extract a single DICOM from the archive and read the SUV factor from it."""
    work = Path(tempfile.mkdtemp(prefix="suv_", dir=tmp_dir))
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = sorted(m for m in z.namelist() if m.lower().endswith(".dcm"))
            if not members:
                return None, ["no .dcm member in archive"], {}
            z.extract(members[0], work)
            r = suv_factor_from_file(work / members[0])
            return r.factor, r.problems, r.detail
    except Exception as e:  # unreadable archive
        return None, [f"archive unreadable: {e}"], {}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process(args_tuple) -> dict:
    scan_dir, data_root, raw_root, tmp_dir, overwrite = args_tuple
    scan_dir = Path(scan_dir)
    rel = scan_dir.relative_to(data_root).as_posix()
    out = scan_dir / "SUV.nii.gz"

    zip_path = raw_archive_for(scan_dir, Path(data_root), Path(raw_root))
    if zip_path is None:
        return {"scan": rel, "status": "no_archive", "problems": "PT.zip not found"}

    factor, problems, detail = factor_for_archive(zip_path, Path(tmp_dir))
    row = {
        "scan": rel,
        "factor": factor if factor is not None else "",
        "units": detail.get("units", ""),
        "decay_correction": detail.get("decay_correction", ""),
        "weight_kg": detail.get("weight_kg", ""),
        "dose_bq": detail.get("dose_bq", ""),
        "half_life_s": detail.get("half_life_s", ""),
        "uptake_delay_s": detail.get("uptake_delay_s", ""),
        "decayed_dose_bq": detail.get("decayed_dose_bq", ""),
        "problems": "; ".join(problems),
    }
    if factor is None:
        row["status"] = "tag_problem"
        return row

    if out.exists() and not overwrite:
        row["status"] = "already"
        return row

    pet_img = sitk.ReadImage(str(scan_dir / "PET.nii.gz"))
    suv = sitk.GetArrayFromImage(pet_img).astype(np.float32) * np.float32(factor)
    suv_img = sitk.GetImageFromArray(suv)
    suv_img.CopyInformation(pet_img)          # identical grid, by construction
    sitk.WriteImage(suv_img, str(out))

    row["status"] = "ok"
    row["suv_max"] = float(suv.max())
    row["suv_p99_9"] = float(np.percentile(suv[suv > 0], 99.9)) if (suv > 0).any() else 0.0
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--tmp-dir", type=Path, default=Path(tempfile.gettempdir()))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    scans = sorted(pet.parent for pet in args.data_root.rglob("PET.nii.gz"))
    if args.limit:
        scans = scans[: args.limit]
    print(f"{len(scans)} scans under {args.data_root}\n")

    work = [(str(s), str(args.data_root), str(args.raw_root), str(args.tmp_dir), args.overwrite)
            for s in scans]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, row in enumerate(pool.map(process, work, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"  {i}/{len(scans)}")

    fields = ["scan", "status", "factor", "units", "decay_correction", "weight_kg", "dose_bq",
              "half_life_s", "uptake_delay_s", "decayed_dose_bq", "suv_max", "suv_p99_9",
              "problems"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\n=== Summary ===")
    for k, v in sorted(by_status.items()):
        print(f"{k:14s} {v}")

    bad = [r for r in rows if r["status"] in ("tag_problem", "no_archive")]
    if bad:
        print(f"\n{len(bad)} scan(s) with missing or odd tags -- NOT converted:")
        for r in bad[:20]:
            print(f"   {r['scan']}: {r['problems']}")

    good = [float(r["factor"]) for r in rows if r.get("factor") not in ("", None)]
    if good:
        a = np.array(good)
        print(f"\nSUV factor over {len(a)} scans: "
              f"mean {a.mean():.4g}, sd {a.std():.4g}, CV {a.std()/a.mean():.1%}")
        p5, p50, p95 = np.percentile(a, [5, 50, 95])
        print(f"  p5 {p5:.4g} | p50 {p50:.4g} | p95 {p95:.4g} | p95/p5 {p95/p5:.2f}x")
    maxima = [float(r["suv_max"]) for r in rows if r.get("suv_max") not in ("", None)]
    if maxima:
        a = np.array(maxima)
        print(f"SUV max per scan: p5 {np.percentile(a,5):.1f} | median {np.median(a):.1f} "
              f"| p95 {np.percentile(a,95):.1f} | overall max {a.max():.1f}")
    print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
