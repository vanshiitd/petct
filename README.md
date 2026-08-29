# MAE training for PET-CT

Implementation and pre-trained weights for 3D multimodal MAE pre-trained models
for PET-CT medical image analysis.

The framework uses an independent-masking masked autoencoder (MAE) pre-training
strategy with a weighted global mean squared error (MSE) loss, combined with
zero (mean) imputation to avoid blocky artifacts. The pre-trained models support
fine-tuning for downstream tasks such as automated tumor segmentation.

> **Note on structure:** the original Jupyter notebooks were converted into the
> `petct/` package and `scripts/` entrypoints below. See
> [`CONVERSION_NOTES.md`](CONVERSION_NOTES.md) for the notebook → command mapping,
> the bugs fixed during conversion, and what was deliberately left unchanged.
> The notebooks remain in git history at commit `1e9d31b`.

## Repository Structure

```
petct/                  # the implementation, imported by the scripts
  config.py             #   architectures, paths, hyperparameters
  splits.py             #   dataset indexing and patient-level train/test split
  data.py               #   MONAI dataloaders
  transforms.py         #   AutoPET preprocessing transform
  volume_ops.py         #   resample / crop / normalize (numpy only)
  models.py             #   backbones, MAE wrapper, weight transfer
  pretrain.py           #   stage 1 training loop
  finetune.py           #   stage 2 training loop + evaluation
  devices.py            #   device selection, AMP, memory helpers

scripts/                # command-line entrypoints
  preprocess_autopet.py
  preprocess_deeppsma.py
  preprocess_spade.py
  extract_vimedpet.py
  scan_dimensions.py
  pretrain.py
  finetune.py

downstreamsplit/        # reference split CSVs
sample_data/            # small local dataset (if present)
weights/                # pre-trained checkpoints (downloaded separately)
runs/                   # fine-tuning outputs
```

## Supported architectures

Selected with `--arch`:

| `--arch` | Architecture | Parameters | Pre-trained checkpoint |
| --- | --- | ---: | --- |
| `small` | SwinUNETRv2-Small `(2,2,2,2)-24` | 18.3M | `swin_small_mae_best.pth` |
| `base` | SwinUNETRv2-Base `(2,2,6,2)-48` | 74.6M | `swin_mae_best_v2.pth` |
| `large` | SwinUNETRv2-Large `(2,2,18,2)-96` | 319.2M | `swin_large_mae_best.pth` |
| `nnunet` | nnU-Net v2 `PlainConvUNet` (3d_fullres) | ~31.2M | `nnunet_v2_mim_best.pth` |

## Pre-Trained Foundation Models

Checkpoints are hosted externally due to GitHub file size limits. Download and
place them in `./weights/`.

**Download link:**
[Pre-Trained Weights (Dropbox)](https://www.dropbox.com/scl/fo/2qojv3vn3nxu6smisot8y/ALAYC8EiLp26TbTTy5DGI08?rlkey=x0620mbaggdq5zi5z2au3fvp4&st=c1uscnzh&dl=0)

## Installation

```bash
conda create -n petct_fm python=3.10
conda activate petct_fm

# Core deep learning and medical imaging libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai einops SimpleITK scipy numpy tqdm nibabel

# Only needed for --arch nnunet
pip install dynamic-network-architectures
```

## Configuration

Paths default to the layout above and can be overridden by environment variable
or per-command flag:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PETCT_AUTOPET_ROOT` | `./sample_data` | labelled AutoPET data |
| `PETCT_PRETRAIN_ROOT` | `./PETCTfoundation` | unlabelled `.npy` corpus |
| `PETCT_WEIGHTS_DIR` | `./weights` | pre-trained checkpoints |
| `PETCT_OUTPUT_DIR` | `./runs` | fine-tuning outputs |

## Usage Guide

Every command supports `--help`.

### 1. Data preprocessing

Handles varying voxel spacings and baseline shifts across AutoPET, DeepPSMA,
SPADE and ViMedPET. Crops background using CT thresholding (`CT > -500`),
standardizes voxel spacing, and independently z-score normalizes PET and CT.

```bash
python scripts/preprocess_autopet.py  --source /data/AutoPET2025_FDG --target /data/PETCTfoundation/AutoPET
python scripts/preprocess_deeppsma.py --source /data/DeepPSMA        --target /data/PETCTfoundation/DeepPSMA
python scripts/preprocess_spade.py    --source /data/spade           --target /data/PETCTfoundation/Spade
python scripts/extract_vimedpet.py    --source /data/ViMedPET        --target /data/unzipViMedPET

# sanity-check the resulting volume dimensions
python scripts/scan_dimensions.py --root /data/PETCTfoundation --exclude-anomalies
```

### 2. Pre-training (masked autoencoder)

Masks 50% of the PET and CT inputs independently and reconstructs the original
volumes.

```bash
python scripts/pretrain.py --arch base --data-root /data/PETCTfoundation
```

### 3. Downstream fine-tuning (segmentation)

Supports checkpoint resumption and cosine annealing. `--init foundation` loads
the pre-trained encoder; `--init scratch` is the random-initialization control.

```bash
# foundation-initialized, all labels
python scripts/finetune.py --arch base --fraction 1.0 --init foundation

# the scratch comparison arm at 10% labels
python scripts/finetune.py --arch base --fraction 0.1 --init scratch

# quick local smoke test on a small dataset
python scripts/finetune.py --arch base --fraction 1.0 --init foundation \
    --split sample --epochs 2 --batch-size 1 --val-interval 1
```

The full label-efficiency experiment is six runs:

```bash
for f in 0.1 0.3 1.0; do
  for init in foundation scratch; do
    python scripts/finetune.py --arch base --fraction $f --init $init
  done
done
```

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--device` | `auto` (default) \| `cpu` \| `cuda` \| `cuda:N` \| `mps` |
| `--split` | `full` (200-patient test set) \| `sample` (small datasets) |
| `--batch-size` | lower if training runs out of memory |
| `--sw-batch-size` | lower if *evaluation* runs out of memory |
| `--seed` | seed Python/NumPy/torch for reproducible runs |

## Utilities

```bash
python view_scan.py PETCT_1bb48bfb40        # render PET / CT / tumour mask
python download_all_patients.py ./AutoPET   # fetch the full TCIA collection
```

## Citation

```bibtex
% Citation details will be added upon publication
```

## Contact

For questions regarding the code, data preprocessing, or model weights, please
open an issue in this repository.
