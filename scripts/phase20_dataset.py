#!/usr/bin/env python3
"""Phase XX: build Dataset507_PSMA_A from A's Phase XVIII exports, and its splits.

See `reports/phase20_segmenter_selfimprove_plan.md`. Images and labels are hard
links to the Dataset505 `imagesTs` / `labelsTs_prep` files Phase XVIII wrote, so
the inputs are byte-identical and cost no disk. Only A patients are linked; B
and C are never placed in a training dataset.

Plans and fingerprint are copied from Dataset505 (`dataset_name` rewritten) so
the network, patch and spacing are the FDG model's. The 5 folds use the same
patient-to-fold map as Phase XIX's detector cross-fitting.

Run, then `nnUNetv2_preprocess -d 507 -plans_name nnUNetPlans_MAE -c 3d_fullres`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SRC = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
SRC_PRE = Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2")
NAME = "Dataset507_PSMA_A"
RAW = Path(r"C:\nnunet_raw") / NAME
PRE = Path(r"C:\nnunet_preprocessed") / NAME
SPLIT = Path(__file__).resolve().parent.parent / "reports" / "phase19_split.json"
SEED, NFOLD = 2027, 5


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.link(src, dst)


def main() -> None:
    A = sorted(json.loads(SPLIT.read_text())["A"])
    (RAW / "imagesTr").mkdir(parents=True, exist_ok=True)
    (RAW / "labelsTr").mkdir(parents=True, exist_ok=True)
    for c in A:
        for ch in ("0000", "0001"):
            link(SRC / "imagesTs" / f"{c}_{ch}.nii.gz", RAW / "imagesTr" / f"{c}_{ch}.nii.gz")
        link(SRC / "labelsTs_prep" / f"{c}.nii.gz", RAW / "labelsTr" / f"{c}.nii.gz")

    ds = json.loads((SRC / "dataset.json").read_text())
    ds.update(numTraining=len(A), dataset_name=NAME,
              description="PSMA-PET-CT-Lesions, Phase XIX split A only, prepared exactly "
                          "as Dataset505 (Phase XVIII export). Phase XX fine-tuning.")
    (RAW / "dataset.json").write_text(json.dumps(ds, indent=1))

    PRE.mkdir(parents=True, exist_ok=True)
    (PRE / "dataset.json").write_text(json.dumps(ds, indent=1))  # preprocess reads it here
    plans = json.loads((SRC_PRE / "nnUNetPlans_MAE.json").read_text())
    plans["dataset_name"] = NAME
    (PRE / "nnUNetPlans_MAE.json").write_text(json.dumps(plans, indent=1))
    (PRE / "dataset_fingerprint.json").write_text((SRC_PRE / "dataset_fingerprint.json").read_text())

    # same map as phase19_selfimprove.py: fold_of = i % 5 over a seed-2027 permutation
    rng = np.random.default_rng(SEED)
    fold_of = {c: i % NFOLD for i, c in enumerate(rng.permutation(A))}
    splits = [{"train": sorted(c for c in A if fold_of[c] != k),
               "val": sorted(c for c in A if fold_of[c] == k)} for k in range(NFOLD)]
    (PRE / "splits_final.json").write_text(json.dumps(splits, indent=1))
    print(f"{NAME}: {len(A)} cases | val sizes {[len(s['val']) for s in splits]}")


if __name__ == "__main__":
    main()
