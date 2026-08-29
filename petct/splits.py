"""Dataset indexing and train/test partitioning.

Deliberately free of torch/MONAI imports: this is pure bookkeeping, it is the
part most worth testing, and keeping it importable without a deep-learning
stack means the split can be inspected on any machine.
"""
from __future__ import annotations

import random
from pathlib import Path

from . import config

# Seeds from the original notebooks; kept so a given dataset partitions the
# same way it did before.
SPLIT_SEED = 42
SUBSET_SEED = 1024


def build_subject_index(base_path: Path) -> dict[str, list[dict]]:
    """Map subject_id -> list of scans, each a dict of the three file paths.

    Results are sorted. The notebooks iterated `rglob` directly, whose order is
    filesystem-dependent, so their "seeded" split was in fact NOT reproducible
    across machines. Sorting makes it genuinely deterministic.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise SystemExit(
            f"Dataset root not found: {base_path}\n"
            f"Set PETCT_AUTOPET_ROOT or pass --data-root."
        )

    subject_dict: dict[str, list[dict]] = {}
    for pet_path in sorted(base_path.rglob("PET.nii.gz")):
        scan_dir = pet_path.parent
        ct_path = scan_dir / "CT_resample.nii.gz"
        seg_path = scan_dir / "tumorSeg.nii.gz"
        if not (ct_path.exists() and seg_path.exists()):
            continue
        subject_id = pet_path.relative_to(base_path).parts[0]
        subject_dict.setdefault(subject_id, []).append(
            {
                "pet_path": str(pet_path),
                "ct_path": str(ct_path),
                "seg_path": str(seg_path),
            }
        )

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
