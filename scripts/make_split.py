#!/usr/bin/env python3
"""Freeze ONE patient-level train/val/test split for every future experiment.

Why this exists: the two earlier runs used different test sets (only ~69 of 200
test patients in common) and selected their best epoch on the test set itself.
Neither number can be compared with the other, or with anything published. From
here on every model trains, validates and is scored on the partition written by
this script.

Rules:
  * The unit is the **patient**. Every scan of a patient lands in one set, so a
    repeat visit can never put the same anatomy in both train and test.
  * **test** = 200 patients drawn only from single-study patients, so multi-scan
    patients (who would otherwise dominate the held-out set) all go to training,
    and each test patient contributes exactly one scan.
  * **val** ~ 10% of the remaining patients, used for checkpoint selection.
  * Both are stratified on tumour presence, so the positive fraction of each set
    matches the dataset as a whole. Without this a 200-patient draw can easily
    land several points off, and Dice over tumour-positive cases would not be
    comparable between splits.
  * A patient counts as tumour-positive when any of its scans has a foreground
    voxel in tumorSeg.

Output: splits/autopet_v1.json -- seed, counts, per-set patient and case IDs.
Case IDs are <PatientID> for a single-study patient and
<PatientID>_<study_folder> for a multi-study one, matching the nnU-Net export.

    python scripts/make_split.py --data-root D:\\data\\autopet_nifti \\
        --out splits/autopet_v1.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np

SEED = 2026
N_TEST = 200
VAL_FRACTION = 0.10


def index_scans(data_root: Path) -> dict[str, list[dict]]:
    """patient -> [{case_id, rel_dir, seg_path}], for every complete scan."""
    patients: dict[str, list[dict]] = {}
    for seg in sorted(data_root.rglob("tumorSeg.nii.gz")):
        scan_dir = seg.parent
        if not ((scan_dir / "PET.nii.gz").exists() and (scan_dir / "CT_resample.nii.gz").exists()):
            continue
        rel = scan_dir.relative_to(data_root)
        patient = rel.parts[0]
        study = "/".join(rel.parts[1:])
        case_id = patient if not study else f"{patient}_{study.replace('/', '_')}"
        patients.setdefault(patient, []).append(
            {"case_id": case_id, "rel_dir": rel.as_posix(), "seg_path": str(seg)}
        )
    return patients


def tumour_positive(seg_path: str) -> bool:
    """True if the mask has any foreground voxel.

    nibabel rather than SimpleITK: ITK's reader segfaulted (0xC0000005) part-way
    through this same sweep while the disk was busy, and nibabel read all of the
    masks without trouble.
    """
    arr = np.asanyarray(nib.load(seg_path).dataobj)
    return bool(np.any(arr > 0))


def stratified_draw(positives: list[str], negatives: list[str], n: int,
                    rng: random.Random) -> list[str]:
    """Draw n patients keeping the positive:negative ratio of the pool."""
    pool_total = len(positives) + len(negatives)
    if n >= pool_total:
        return sorted(positives + negatives)
    n_pos = round(n * len(positives) / pool_total)
    n_pos = min(n_pos, len(positives))
    n_neg = min(n - n_pos, len(negatives))
    n_pos = n - n_neg  # if negatives ran short, take the balance from positives
    return sorted(rng.sample(positives, n_pos) + rng.sample(negatives, n_neg))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("splits/autopet_v1.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--cache", type=Path, default=None,
                   help="JSON cache of per-scan tumour flags (reading ~1000 masks is slow)")
    args = p.parse_args()

    patients = index_scans(args.data_root)
    if not patients:
        raise SystemExit(f"No complete scans under {args.data_root}")
    n_scans = sum(len(v) for v in patients.values())
    print(f"Indexed {len(patients)} patients / {n_scans} scans")

    cache: dict[str, bool] = {}
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text())

    print("Reading masks for tumour presence …")
    for i, (pid, scans) in enumerate(sorted(patients.items()), 1):
        for scan in scans:
            key = scan["case_id"]
            if key not in cache:
                cache[key] = tumour_positive(scan["seg_path"])
            scan["tumour_positive"] = cache[key]
        if i % 100 == 0:
            print(f"  {i}/{len(patients)}")
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, indent=1))

    pos_patients = {pid for pid, scans in patients.items()
                    if any(s["tumour_positive"] for s in scans)}
    overall_pos = len(pos_patients) / len(patients)
    print(f"tumour-positive patients: {len(pos_patients)}/{len(patients)} ({overall_pos:.1%})")

    single = sorted(pid for pid, scans in patients.items() if len(scans) == 1)
    multi = sorted(pid for pid, scans in patients.items() if len(scans) > 1)
    print(f"single-study patients: {len(single)} | multi-study: {len(multi)}")

    rng = random.Random(args.seed)
    test = stratified_draw([p for p in single if p in pos_patients],
                           [p for p in single if p not in pos_patients],
                           args.n_test, rng)

    remaining = sorted(set(patients) - set(test))
    n_val = round(len(remaining) * args.val_fraction)
    val = stratified_draw([p for p in remaining if p in pos_patients],
                          [p for p in remaining if p not in pos_patients],
                          n_val, rng)
    train = sorted(set(remaining) - set(val))

    # --- the split must be a partition -----------------------------------
    sets = {"train": train, "val": val, "test": test}
    for a in sets:
        for b in sets:
            if a < b:
                overlap = set(sets[a]) & set(sets[b])
                assert not overlap, f"patient in both {a} and {b}: {sorted(overlap)[:5]}"
    assert set(train) | set(val) | set(test) == set(patients), "split does not cover every patient"
    assert len(train) + len(val) + len(test) == len(patients), "duplicate patient in the split"
    assert all(len(patients[p]) == 1 for p in test), "test set must hold only single-scan patients"
    print("OK: partition verified (no patient in two sets, every patient assigned)")

    def summarise(name: str, pids: list[str]) -> dict:
        scans = [s for p in pids for s in patients[p]]
        n_pos_scans = sum(1 for s in scans if s["tumour_positive"])
        n_pos_pat = sum(1 for p in pids if p in pos_patients)
        print(f"{name:5s}: {len(pids):3d} patients | {len(scans):3d} scans | "
              f"tumour-positive {n_pos_pat} patients ({n_pos_pat/max(1,len(pids)):.1%}), "
              f"{n_pos_scans} scans")
        return {
            "n_patients": len(pids),
            "n_scans": len(scans),
            "n_tumour_positive_patients": n_pos_pat,
            "n_tumour_positive_scans": n_pos_scans,
            "tumour_positive_patient_fraction": round(n_pos_pat / max(1, len(pids)), 4),
            "patients": pids,
            "cases": {p: [s["case_id"] for s in patients[p]] for p in pids},
        }

    payload = {
        "name": args.out.stem,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "unit": "patient",
        "rules": {
            "test": f"{args.n_test} patients, single-study only, stratified on tumour presence",
            "val": f"{args.val_fraction:.0%} of the remaining patients, stratified the same way",
            "train": "every remaining patient",
        },
        "totals": {
            "n_patients": len(patients),
            "n_scans": n_scans,
            "n_multi_scan_patients": len(multi),
            "n_tumour_positive_patients": len(pos_patients),
            "tumour_positive_patient_fraction": round(overall_pos, 4),
        },
        "splits": {name: summarise(name, pids) for name, pids in
                   (("train", train), ("val", val), ("test", test))},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
