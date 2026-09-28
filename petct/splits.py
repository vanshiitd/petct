"""Dataset indexing and train/test partitioning.

Deliberately free of torch/MONAI imports: this is pure bookkeeping, it is the
part most worth testing, and keeping it importable without a deep-learning
stack means the split can be inspected on any machine.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config

# Fixed seeds so a given dataset always partitions the same way.
SPLIT_SEED = 42
SUBSET_SEED = 1024


def build_subject_index(base_path: Path) -> dict[str, list[dict]]:
    """Map subject_id -> list of scans, each a dict of the three file paths.

    Results are sorted. `rglob` alone returns filesystem-dependent order, which
    would make the seeded split below non-reproducible across machines; sorting
    is what makes the seeds actually deterministic.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise SystemExit(
            f"Dataset root not found: {base_path}\n"
            f"Set PETCT_AUTOPET_ROOT or pass --data-root."
        )

    subject_dict: dict[str, list[dict]] = {}
    excluded = 0
    for pet_path in sorted(base_path.rglob("PET.nii.gz")):
        scan_dir = pet_path.parent
        ct_path = scan_dir / "CT_resample.nii.gz"
        seg_path = scan_dir / "tumorSeg.nii.gz"
        if not (ct_path.exists() and seg_path.exists()):
            continue
        subject_id = pet_path.relative_to(base_path).parts[0]
        if any(bad in subject_id for bad in config.SEGMENTATION_ANOMALIES):
            excluded += 1
            continue
        subject_dict.setdefault(subject_id, []).append(
            {
                "pet_path": str(pet_path),
                "ct_path": str(ct_path),
                "seg_path": str(seg_path),
            }
        )

    if excluded:
        print(f"Excluding {excluded} scan(s) listed in config.SEGMENTATION_ANOMALIES.")

    if not subject_dict:
        raise SystemExit(
            f"No usable scans under {base_path}. Each scan folder needs "
            f"PET.nii.gz, CT_resample.nii.gz and tumorSeg.nii.gz."
        )
    return subject_dict


def split_subjects(
    subject_dict: dict[str, list[dict]],
    train_fraction: float,
    split: config.SplitPreset,
) -> tuple[list[dict], list[dict]]:
    """Partition subjects into (train_files, test_files).

    The split is at *patient* level, never scan level -- the same patient's
    anatomy appearing in both train and test would inflate apparent accuracy.

    Test subjects are drawn only from single-scan patients, so multi-scan
    patients (which would otherwise dominate the held-out set) all go to train.
    """
    single = [s for s, scans in subject_dict.items() if len(scans) == 1]
    multi = [s for s, scans in subject_dict.items() if len(scans) > 1]

    random.seed(SPLIT_SEED)
    random.shuffle(single)
    random.shuffle(multi)

    n_test = min(split.n_test, len(single))
    test_subjects = single[:n_test]
    remaining_singles = single[n_test:]

    if split.max_multiscan_train is not None:
        # Fill from multi-scan patients up to the scan cap, then top up with
        # single-scan patients until the pool reaches train_pool_scans.
        base_train: list[str] = []
        count = 0
        for sub in multi:
            n = len(subject_dict[sub])
            if count + n <= split.max_multiscan_train:
                base_train.append(sub)
                count += n
        needed = (split.train_pool_scans or 0) - count
        base_train.extend(remaining_singles[:max(0, needed)])
        pool_scans = split.train_pool_scans or sum(len(subject_dict[s]) for s in base_train)
    else:
        # Small-dataset preset: everything not held out for test is trainable.
        base_train = multi + remaining_singles
        pool_scans = sum(len(subject_dict[s]) for s in base_train)

    target_scans = max(1, int(pool_scans * train_fraction))

    random.seed(SUBSET_SEED)
    random.shuffle(base_train)

    final_train, count = [], 0
    for sub in base_train:
        if count >= target_scans:
            break
        final_train.append(sub)
        count += len(subject_dict[sub])

    train_files = [scan for s in final_train for scan in subject_dict[s]]
    test_files = [scan for s in test_subjects for scan in subject_dict[s]]
    return train_files, test_files


def load_frozen_split(
    subject_dict: dict[str, list[dict]],
    split_file: Path,
    train_fraction: float = 1.0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read the frozen patient-level split written by scripts/make_split.py.

    Returns (train_files, val_files, test_files). The partition itself is fixed
    on disk, so unlike `split_subjects` nothing here depends on a seed or on the
    order the filesystem returns; every model sees exactly the same patients.

    `train_fraction` still subsets the *training* patients for label-efficiency
    experiments, seeded so that a given fraction is reproducible and nested (a
    smaller fraction is a subset of a larger one). Val and test never change.
    """
    split_file = Path(split_file)
    if not split_file.exists():
        raise SystemExit(f"Split file not found: {split_file}")
    payload = json.loads(split_file.read_text())

    def files_for(names: list[str], what: str) -> list[dict]:
        missing = [n for n in names if n not in subject_dict]
        if missing:
            raise SystemExit(
                f"{len(missing)} {what} patient(s) from {split_file.name} are absent from the "
                f"dataset, e.g. {missing[:5]}. The split and the data root disagree."
            )
        return [scan for n in names for scan in subject_dict[n]]

    train_patients = list(payload["splits"]["train"]["patients"])
    val_files = files_for(payload["splits"]["val"]["patients"], "val")
    test_files = files_for(payload["splits"]["test"]["patients"], "test")

    if train_fraction < 1.0:
        rng = random.Random(SUBSET_SEED)
        rng.shuffle(train_patients)
        target = max(1, int(round(len(train_patients) * train_fraction)))
        train_patients = train_patients[:target]
    train_files = files_for(sorted(train_patients), "train")

    return train_files, val_files, test_files
