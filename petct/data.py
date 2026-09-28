"""Dataloader construction for pretraining and fine-tuning.

The indexing/splitting logic lives in splits.py (which needs no torch/MONAI);
this module only wraps it in MONAI Datasets and DataLoaders.
"""
from __future__ import annotations

from pathlib import Path

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, RandSpatialCropd, RandZoomd, SpatialPadd, ToTensord,
)

from . import config
from .splits import (  # re-exported for convenience
    build_subject_index, load_frozen_split, split_subjects,
)
from .transforms import AutoPETPreprocessd

__all__ = [
    "build_subject_index",
    "split_subjects",
    "load_frozen_split",
    "get_segmentation_dataloaders",
    "get_pretraining_dataloader",
]


# ---------------------------------------------------------------------------
# Segmentation (fine-tuning) data
# ---------------------------------------------------------------------------
def get_segmentation_dataloaders(
    data_root: Path,
    train_fraction: float = 1.0,
    batch_size: int = 4,
    split_name: str = "full",
    num_workers: int = 4,
    pin_memory: bool = True,
    split_file: Path | None = None,
):
    """Build the dataloaders for fine-tuning.

    Returns (train_loader, val_loader, test_loader). With `split_file` the
    partition comes from the frozen split on disk and `val_loader` is a real,
    separate set for checkpoint selection; without it the legacy seeded split is
    used and `val_loader` is None, leaving the caller to select on test as
    before.
    """
    subject_dict = build_subject_index(data_root)
    n_scans = sum(len(v) for v in subject_dict.values())
    print(f"Parsed {len(subject_dict)} patients / {n_scans} scans from {data_root}")

    val_files: list[dict] = []
    if split_file is not None:
        train_files, val_files, test_files = load_frozen_split(
            subject_dict, split_file, train_fraction)
        print(
            f"=== frozen split '{Path(split_file).name}'  "
            f"train_fraction={train_fraction:.0%} ===\n"
            f"Train: {len(train_files)} scans | Val: {len(val_files)} scans | "
            f"Test: {len(test_files)} scans"
        )
    else:
        split = config.get_split(split_name)
        train_files, test_files = split_subjects(subject_dict, train_fraction, split)
        print(
            f"=== split='{split_name}'  train_fraction={train_fraction:.0%} ===\n"
            f"Train: {len(train_files)} scans | Test: {len(test_files)} scans"
        )
    if not train_files:
        raise SystemExit("Training split is empty -- check --train-fraction and --split.")

    train_tf = Compose([
        AutoPETPreprocessd(keys=["image", "label"]),
        # A few patients' body-cropped volumes are smaller than ROI_SIZE in at
        # least one dimension (limited-coverage acquisitions). RandSpatialCropd
        # with random_size=False needs the input to be at least the crop size,
        # so pad first; zero is the dataset mean after z-scoring, and the
        # padding is dropped again by the crop whenever the volume was already
        # large enough.
        SpatialPadd(keys=["image", "label"], spatial_size=config.ROI_SIZE,
                    mode="constant", constant_values=0),
        RandSpatialCropd(keys=["image", "label"], roi_size=config.ROI_SIZE, random_size=False),
        ToTensord(keys=["image", "label"]),
    ])
    test_tf = Compose([
        AutoPETPreprocessd(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

    train_loader = DataLoader(
        Dataset(data=train_files, transform=train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # Evaluation always at batch_size 1: volumes are full-size and vary in
    # shape, so they cannot be collated into a batch.
    def eval_loader(files: list[dict]):
        return DataLoader(
            Dataset(data=files, transform=test_tf),
            batch_size=1, shuffle=False, num_workers=num_workers,
        )

    return train_loader, (eval_loader(val_files) if val_files else None), eval_loader(test_files)


# ---------------------------------------------------------------------------
# Pretraining data
# ---------------------------------------------------------------------------
def get_pretraining_dataloader(
    data_root: Path,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    """Dataloader over the unlabelled .npy corpus used for MAE pretraining.

    Each file is a preprocessed (2, Z, Y, X) volume produced by one of the
    scripts/preprocess_*.py pipelines.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise SystemExit(
            f"Pretraining corpus not found: {data_root}\n"
            f"Set PETCT_PRETRAIN_ROOT or pass --data-root."
        )

    data_dicts = []
    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for f in sorted(dataset_dir.rglob("*.npy")):
            if any(bad in f.as_posix() for bad in config.PRETRAIN_ANOMALIES):
                continue
            data_dicts.append({"image": f.as_posix()})

    if not data_dicts:
        raise SystemExit(f"No .npy volumes found under {data_root}.")
    print(f"Total valid 3D volumes loaded: {len(data_dicts)}")

    train_tf = Compose([
        LoadImaged(keys=["image"], reader="NumpyReader"),
        RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1,
                  mode="trilinear", keep_size=False),
        SpatialPadd(keys=["image"], spatial_size=config.ROI_SIZE,
                    mode="constant", constant_values=0),
        RandSpatialCropd(keys=["image"], roi_size=config.ROI_SIZE, random_size=False),
        ToTensord(keys=["image"]),
    ])

    return DataLoader(
        Dataset(data=data_dicts, transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
