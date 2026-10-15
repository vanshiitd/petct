#!/usr/bin/env python3
"""Add 5 patient-grouped cross-validation folds over the training scans.

The 738 training scans come from 630 patients, and a patient's repeat visits
are highly correlated, so folds split on patient, never on scan. Stratified on
whether a patient has any tumour-positive scan, so each fold keeps the parent
split's positive rate instead of drifting with the draw.

These become folds 3..7 of Dataset505's `splits_final.json`. Folds 0-2 are the
existing 100%, 10% and 30% entries and must survive untouched: an earlier
version of `add_subset_fold.py` spliced rather than appended and silently
destroyed a fold that had already been trained, so every pre-existing fold is
compared byte-for-byte before anything is written.

Each new fold trains on four fifths of the training scans and validates on the
remaining fifth. That held-out fifth is what makes out-of-fold predictions
possible for every training scan: scan s is predicted by the one fold model
that never saw s. The val (76) and test (200) sets are not touched by any of
this.

    python scripts/make_cv_folds.py --splits-final <...>/splits_final.json \\
        --split splits/autopet_v1.json --n-folds 5 --seed 20261008
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-final", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(r"C:\data\autopet_nifti_v2"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folds = json.loads(args.splits_final.read_text())
    before = [json.dumps(f, sort_keys=True) for f in folds]
    n_before = len(folds)
    print(f"{n_before} existing fold(s): "
          + ", ".join(f"{i}: {len(f['train'])}/{len(f['val'])}"
                      for i, f in enumerate(folds)))

    split = json.loads(args.split.read_text())
    cases = split["splits"]["train"]["cases"]        # patient -> [scan ids]
    all_scans = sorted(s for ss in cases.values() for s in ss)

    import SimpleITK as sitk

    def scan_dir(case_id: str) -> Path:
        d = args.root / case_id
        if (d / "tumorSeg.nii.gz").exists():
            return d
        parts = case_id.split("_")
        patient = f"{parts[0]}_{parts[1]}"
        return args.root / patient / case_id[len(patient) + 1:]

    positive = set()
    for i, s in enumerate(all_scans, 1):
        if sitk.GetArrayFromImage(sitk.ReadImage(
                str(scan_dir(s) / "tumorSeg.nii.gz"))).any():
            positive.add(s)
        if i % 200 == 0:
            print(f"  scanned {i}/{len(all_scans)}", flush=True)
    if len(positive) != split["splits"]["train"]["n_tumour_positive_scans"]:
        raise SystemExit(f"found {len(positive)} tumour-positive scans, split file "
                         f"says {split['splits']['train']['n_tumour_positive_scans']}")

    pos_pat = sorted(p for p, ss in cases.items() if any(s in positive for s in ss))
    neg_pat = sorted(p for p in cases if p not in set(pos_pat))
    rng = np.random.default_rng(args.seed)
    print(f"\n{len(cases)} patients: {len(pos_pat)} tumour-positive, {len(neg_pat)} free")

    # deal each stratum round-robin after shuffling: equal sizes without
    # letting a stratum pile into one fold
    assign: dict[str, int] = {}
    for stratum in (pos_pat, neg_pat):
        order = [stratum[i] for i in rng.permutation(len(stratum))]
        for k, p in enumerate(order):
            assign[p] = k % args.n_folds

    new = []
    print(f"\n{'fold':>5s} {'patients':>9s} {'train scans':>12s} {'val scans':>10s} "
          f"{'val pos':>8s} {'pos frac':>9s}")
    for k in range(args.n_folds):
        val_pat = [p for p, f in assign.items() if f == k]
        val_scans = sorted(s for p in val_pat for s in cases[p])
        train_scans = sorted(s for s in all_scans if s not in set(val_scans))
        new.append({"train": train_scans, "val": val_scans})
        npos = sum(1 for s in val_scans if s in positive)
        print(f"{n_before + k:>5d} {len(val_pat):>9d} {len(train_scans):>12d} "
              f"{len(val_scans):>10d} {npos:>8d} {npos / len(val_scans):>9.4f}")

    # every scan held out exactly once -- the property out-of-fold prediction needs
    held = [s for f in new for s in f["val"]]
    if sorted(held) != all_scans:
        raise SystemExit(f"held-out union is {len(set(held))} unique of {len(all_scans)}; "
                         f"duplicates {len(held) - len(set(held))}")
    print(f"\nevery training scan held out exactly once: OK ({len(all_scans)} scans)")
    for k, f in enumerate(new):
        if set(f["train"]) & set(f["val"]):
            raise SystemExit(f"fold {n_before + k} overlaps train and val")
    print("no fold has a scan in both train and val: OK")

    out = folds + new
    after = [json.dumps(f, sort_keys=True) for f in out[:n_before]]
    if after != before:
        bad = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
        raise SystemExit(f"existing fold(s) {bad} changed; refusing to write")
    print(f"all {n_before} existing fold(s) byte-identical: OK")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    args.splits_final.write_text(json.dumps(out, sort_keys=True, indent=False))
    print(f"\nwritten {len(out)} folds to {args.splits_final}")


if __name__ == "__main__":
    main()
