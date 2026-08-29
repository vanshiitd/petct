"""Central configuration: architectures, paths, and training defaults.

Everything that used to be hardcoded in 24 separate notebooks lives here once.
Paths can be overridden with environment variables so the same code runs on a
laptop, a lab workstation, and an HPC node without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


@dataclass
class Paths:
    """Filesystem locations. Override any of these via environment variable."""

    # Labelled segmentation data (AutoPET): one folder per scan containing
    # PET.nii.gz, CT_resample.nii.gz, tumorSeg.nii.gz
    autopet_root: Path = field(
        default_factory=lambda: _env_path("PETCT_AUTOPET_ROOT", "./sample_data")
    )
    # Unlabelled pretraining data: .npy volumes of shape (2, Z, Y, X)
    pretrain_root: Path = field(
        default_factory=lambda: _env_path("PETCT_PRETRAIN_ROOT", "./PETCTfoundation")
    )
    # Where checkpoints are written / read
    weights_dir: Path = field(
        default_factory=lambda: _env_path("PETCT_WEIGHTS_DIR", "./weights")
    )
    # Where per-run outputs (checkpoints, logs) are written
    output_dir: Path = field(
        default_factory=lambda: _env_path("PETCT_OUTPUT_DIR", "./runs")
    )


PATHS = Paths()


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchSpec:
    """One backbone configuration.

    `family` selects the builder; everything else is the per-variant delta that
    used to be the *only* difference between whole duplicated notebooks.
    """

    name: str
    family: str  # "swin" | "nnunet"
    lr: float
    pretrain_ckpt: str  # filename of the MAE/MIM checkpoint this arch produces
    weight_prefix: str  # state_dict prefix to strip when transferring weights
    skip_on_transfer: tuple[str, ...] = ()  # substrings of keys NOT to transfer

    # swin-only
    feature_size: int | None = None
    depths: Sequence[int] | None = None
    num_heads: Sequence[int] = (3, 6, 12, 24)


ARCHS: dict[str, ArchSpec] = {
    "small": ArchSpec(
        name="small", family="swin", feature_size=24, depths=(2, 2, 2, 2),
        lr=1e-4, pretrain_ckpt="swin_mae_best_v2.pth", weight_prefix="encoder_decoder.",
    ),
    "base": ArchSpec(
        name="base", family="swin", feature_size=48, depths=(2, 2, 6, 2),
        lr=1e-4, pretrain_ckpt="swin_mae_best_v2.pth", weight_prefix="encoder_decoder.",
    ),
    "large": ArchSpec(
        name="large", family="swin", feature_size=96, depths=(2, 2, 18, 2),
        lr=1e-4, pretrain_ckpt="swin_mae_best_v2.pth", weight_prefix="encoder_decoder.",
    ),
    # nnU-Net v2 backbone. Higher LR than the transformers, per the original
    # notebook's comment ("CNNs can be slightly more aggressive").
    # The final classification head is deliberately NOT transferred: the
    # pretraining head reconstructs 2 image channels, the segmentation head
    # predicts 2 classes -- same shape, completely different meaning.
    "nnunet": ArchSpec(
        name="nnunet", family="nnunet",
        lr=5e-4, pretrain_ckpt="nnunet_v2_mim_best.pth", weight_prefix="nnunet.",
        skip_on_transfer=("seg_outputs",),
    ),
}


def get_arch(name: str) -> ArchSpec:
    try:
        return ARCHS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown architecture {name!r}. Choose from: {', '.join(ARCHS)}"
        )


# ---------------------------------------------------------------------------
# Data / training defaults
# ---------------------------------------------------------------------------
IN_CHANNELS = 2  # PET, CT
NUM_CLASSES = 2  # background, tumour
ROI_SIZE = (96, 128, 128)  # training crop / sliding-window size
TARGET_SPACING_ZYX = (3.0, 2.0, 2.0)  # mm
CT_BODY_THRESHOLD = -500  # HU; above this is body, below is air/bed
CROP_PAD = 3  # voxels of margin kept around the body bounding box

# MAE pretraining
MASK_PATCH_SIZE = (12, 16, 16)  # voxels per masked block -> 8x8x8 grid over ROI
MASK_RATIO = 0.5
UNMASKED_LOSS_WEIGHT = 0.2  # masked region is weighted 1.0

# Optimisation
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
LR_MIN = 1e-6
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 100
VAL_INTERVAL = 20
SLIDING_WINDOW_OVERLAP = 0.5

# Volumes known to be corrupt in the pretraining corpus; skipped when indexing.
PRETRAIN_ANOMALIES = (
    "LDca4f40/LDca5687", "LDca56d5/LDca5e13", "LDca4eed/LDca54ed",
    "LDca519f/LDca5602", "LDca56da/LDca5d8b", "LDca58c7/LDca5c77",
    "LDca58c3/LDca5c6f", "LDca4f3f/LDca5507", "LDca58c2/LDca5c6e",
    "LDca5163/LDca5581", "LDca56d2/LDca5d2b",
)


# ---------------------------------------------------------------------------
# Train/test split presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitPreset:
    """Controls how patients are partitioned into train/test.

    `full` reproduces the notebooks' original scheme. `sample` is for small
    local datasets (e.g. the 20-patient smoke-test set) where holding out 200
    patients is impossible.
    """

    name: str
    n_test: int
    max_multiscan_train: int | None  # cap on scans drawn from multi-scan patients
    train_pool_scans: int | None  # total scans in the 100% training pool


SPLITS: dict[str, SplitPreset] = {
    "full": SplitPreset("full", n_test=200, max_multiscan_train=400, train_pool_scans=700),
    "sample": SplitPreset("sample", n_test=4, max_multiscan_train=None, train_pool_scans=None),
}


def get_split(name: str) -> SplitPreset:
    try:
        return SPLITS[name]
    except KeyError:
        raise SystemExit(f"Unknown split preset {name!r}. Choose from: {', '.join(SPLITS)}")
