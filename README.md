# MAE training for PET-CT

Implementation and pre-trained weights for 3D multimodal MAE pre-trained models
for PET-CT medical image analysis.

The framework uses an independent-masking masked autoencoder (MAE) pre-training
strategy with a weighted global mean squared error (MSE) loss, combined with
zero (mean) imputation to avoid blocky artifacts. The pre-trained models support
fine-tuning for downstream tasks such as automated tumor segmentation.

See [`DESIGN_NOTES.md`](DESIGN_NOTES.md) for design decisions, memory
behaviour, and known limitations.

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
  download_tcia.py      #   fetch the AutoPET collection from TCIA (raw DICOM)
  dicom_to_nifti.py     #   raw TCIA DICOM -> the NIfTI layout below
  preprocess_autopet.py
  preprocess_deeppsma.py
  preprocess_spade.py
  extract_vimedpet.py
  scan_dimensions.py
  pretrain.py
  finetune.py

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

### 1. Check what GPU you have

```bash
nvidia-smi          # note the CUDA version in the top-right
```

### 2. Create an environment

```bash
conda create -n petct python=3.10 -y
conda activate petct
```

(`python -m venv .venv && source .venv/bin/activate` works equally well.)

### 3. Install PyTorch matched to your CUDA version

Do this **before** the other requirements — a plain `pip install torch` can pull
a CPU-only build, and training will then silently run on CPU.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# or
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
```

Verify before continuing:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` on the second value means the GPU is visible. If it prints `False`, fix
that now rather than after starting a training run.

### 4. Install the rest

```bash
pip install -r requirements.txt
```

That covers everything needed to convert data, train and evaluate. Two optional
extras are listed (commented out) at the bottom of `requirements.txt`:
`matplotlib` for `view_scan.py`, and `dynamic-network-architectures` for
`--arch nnunet`.

### 5. Confirm it works

```bash
python scripts/finetune.py --help
```

If that prints usage rather than a traceback, the install is good.

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

### 0a. Download the data (skip if you already have it)

```bash
python scripts/download_tcia.py /data/AutoPET_raw --limit 20   # trial run
python scripts/download_tcia.py /data/AutoPET_raw              # the full ~419 GB
```

Standard library only, and resumable — rerun the same command to continue after
an interruption. Behind a proxy, set `HTTPS_PROXY` / `HTTP_PROXY` first.

### 0b. Convert the downloaded data

TCIA ships raw DICOM; the training pipeline expects NIfTI. This step bridges
that gap and auto-detects what it is given (DICOM directories, per-patient zip
archives, or already-converted NIfTI):

```bash
python scripts/dicom_to_nifti.py --source /data/AutoPET_raw --target /data/autopet_nifti

# try a few patients first to check it works on your layout
python scripts/dicom_to_nifti.py --source /data/AutoPET_raw --target /data/autopet_nifti --limit 5
```

It produces, per patient: `PET.nii.gz`, `CT_resample.nii.gz`, `tumorSeg.nii.gz`.
Safe to interrupt and rerun — converted patients are skipped.

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
python view_scan.py PETCT_1bb48bfb40   # render PET / CT / tumour mask for one patient
```

`tcia_api.py` provides a small client for TCIA's REST API if you need to script
further downloads.

## Citation

```bibtex
% Citation details will be added upon publication
```

## Contact

For questions regarding the code, data preprocessing, or model weights, please
open an issue in this repository.
