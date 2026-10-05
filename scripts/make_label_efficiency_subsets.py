#!/usr/bin/env python3
"""Carve nested training subsets out of the frozen split, for label-efficiency runs.

Two properties matter and neither is automatic:

  * **Patient-level.** A patient's repeat visits are highly correlated, so a subset
    takes every scan of a chosen patient or none of them. Sampling scans would leak
    a patient's second visit into a subset that is supposed not to know them.
  * **Nested.** The 10% subset must be contained in the 30% one, or the two points
    measure different training sets as well as different sizes. That is arranged by
    ordering each stratum once under a fixed seed and taking prefixes: the first
    10% of an ordering is always inside the first 30% of the same ordering.

Stratified on tumour-positive patient status, so a small subset keeps the parent
split's positive fraction instead of drifting with the draw.

Subsets are written back into the split file under `splits.train.subsets`, leaving
every existing field untouched. Validation and test are never resampled.

    python scripts/make_label_efficiency_subsets.py --split splits/autopet_v1.json \\
        --fraction 0.10 --seed 20261005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def subset_for(fraction: float, order_pos: list[str], order_neg: list[str],
               cases: dict[str, list[str]], positive_scans: set[str]) -> dict:
    """The first `fraction` of each stratum's fixed ordering."""
    n_pos = int(round(fraction * len(order_pos)))
    n_neg = int(round(fraction * len(order_neg)))
    patients = sorted(order_pos[:n_pos] + order_neg[:n_neg])
    picked = {p: cases[p] for p in patients}
    scans = [s for ss in picked.values() for s in ss]
    return {
        "n_patients": len(patients),
        "n_scans": len(scans),
        "n_tumour_positive_patients": n_pos,
        "n_tumour_positive_scans": sum(1 for s in scans if s in positive_scans),
        "tumour_positive_patient_fraction": round(n_pos / max(1, len(patients)), 4),
        "patients": patients,
        "cases": picked,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path(r"C:\data\autopet_nifti_v2"),
                   help="converted data root, read to decide which scans have lesions")
    p.add_argument("--fraction", type=float, action="append", required=True,
                   help="repeatable; each becomes an entry under splits.train.subsets")
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    train = split["splits"]["train"]
    cases: dict[str, list[str]] = train["cases"]

    # which scans carry a lesion, read from the data rather than assumed
    import SimpleITK as sitk

    def scan_dir(case_id: str) -> Path:
        d = args.root / case_id
        if (d / "tumorSeg.nii.gz").exists():
            return d
        parts = case_id.split("_")
        patient = f"{parts[0]}_{parts[1]}"
        return args.root / patient / case_id[len(patient) + 1:]

    positive_scans = set()
    all_scans = [s for ss in cases.values() for s in ss]
    for i, case in enumerate(all_scans, 1):
        f = scan_dir(case) / "tumorSeg.nii.gz"
        if sitk.GetArrayFromImage(sitk.ReadImage(str(f))).any():
            positive_scans.add(case)
        if i % 200 == 0:
            print(f"  scanned {i}/{len(all_scans)}", flush=True)

    if len(positive_scans) != train["n_tumour_positive_scans"]:
        raise SystemExit(f"found {len(positive_scans)} tumour-positive scans but the "
                         f"split says {train['n_tumour_positive_scans']}")

    pos_patients = sorted(p_ for p_, ss in cases.items()
                          if any(s in positive_scans for s in ss))
    neg_patients = sorted(p_ for p_ in cases if p_ not in set(pos_patients))
    print(f"\n{len(cases)} training patients: {len(pos_patients)} tumour-positive, "
          f"{len(neg_patients)} tumour-free")

    rng = np.random.default_rng(args.seed)
    order_pos = [pos_patients[i] for i in rng.permutation(len(pos_patients))]
    order_neg = [neg_patients[i] for i in rng.permutation(len(neg_patients))]

    subsets = train.get("subsets", {})
    subsets["method"] = ("patient-level, stratified on tumour-positive status; each "
                         "stratum is ordered once under `seed` and a k fraction takes "
                         "that ordering's first k, so subsets are nested")
    subsets["seed"] = args.seed

    made = []
    for frac in sorted(args.fraction):
        key = f"{round(frac * 100)}pct"
        s = subset_for(frac, order_pos, order_neg, cases, positive_scans)
        subsets[key] = s
        made.append((key, s))
        print(f"\n{key}: {s['n_patients']} patients, {s['n_scans']} scans, "
              f"{s['n_tumour_positive_scans']} tumour-positive scans "
              f"({s['tumour_positive_patient_fraction']:.4f} positive patients, "
              f"parent {train['tumour_positive_patient_fraction']})")

    # nesting is the property the whole design rests on: check it, do not trust it
    for (ka, sa), (kb, sb) in zip(made, made[1:]):
        missing = set(sa["patients"]) - set(sb["patients"])
        if missing:
            raise SystemExit(f"{ka} is not contained in {kb}: {len(missing)} patient(s) lost")
        print(f"nesting {ka} subset of {kb}: OK")
    for key, s in made:
        stray = set(s["patients"]) - set(cases)
        if stray:
            raise SystemExit(f"{key} contains patients outside the training split: {stray}")
        print(f"{key} entirely inside the training split: OK")

    if args.dry_run:
        print("\n--dry-run: split file not modified")
        return
    train["subsets"] = subsets
    args.split.write_text(json.dumps(split, indent=1))
    print(f"\nwritten to {args.split}")


if __name__ == "__main__":
    main()
