# Notebook → script conversion

The 24 notebooks were converted into a `petct/` package plus `scripts/`
entrypoints. This document records what moved where, what was fixed, and what
was deliberately left alone.

The original `.ipynb` files are untouched — nothing was deleted. You can diff
against them or fall back to them at any time.

---

## 1. Why the code shrank so much

10,469 lines of notebook code became roughly 1,400 lines of package code. Almost
nothing was removed; it was **deduplicated**. The 24 notebooks varied along only
three axes:

| Axis | Values | Lines that actually differed |
|---|---|---|
| Backbone | small / base / large / nnunet | 2 (`feature_size`, `depths`) |
| Label fraction | 10% / 30% / 100% | 1 (`train_fraction`) |
| Initialisation | scratch / mae | 1 (`use_foundation`) |

Every one of those is now a command-line flag. For example
`swinv2large/30%training.ipynb` and `swinv2small/10%training.ipynb` were ~360-line
files differing in four meaningful lines.

---

## 2. Where each notebook went

### Preprocessing

| Notebook | Replacement |
|---|---|
| `preprocessing/autopetproc.ipynb` | `scripts/preprocess_autopet.py` |
| `preprocessing/spadeproc.ipynb` (cell 0) | `scripts/preprocess_spade.py` |
| `preprocessing/spadeproc.ipynb` (cells 1, 2) | `scripts/scan_dimensions.py` |
| `preprocessing/DeepPSMA copy.ipynb` | `scripts/preprocess_deeppsma.py` |
| `preprocessing/ViMedPETproc.ipynb` | `scripts/extract_vimedpet.py` |

Cells 1 and 2 of `spadeproc` were two near-identical copies of the same
dimension scanner; the only difference was that cell 2 filtered out known-bad
volumes. That is now the `--exclude-anomalies` flag.

### Pretraining (Stage 1)

| Notebooks | Replacement |
|---|---|
| `swinv2{small,base,large}/MAEv2.ipynb`, `nnunetv2/nnunetv2MAE.ipynb` | `scripts/pretrain.py --arch {small,base,large,nnunet}` |

### Fine-tuning (Stage 2)

All 18 `{10,30,100}%training[-mae].ipynb` files across the four backbone folders
collapse into one command:

```bash
python scripts/finetune.py --arch <arch> --fraction <f> --init {foundation,scratch}
```

| Old notebook | New command |
|---|---|
| `swinv2base/100%training.ipynb` (mae cell) | `--arch base --fraction 1.0 --init foundation` |
| `swinv2base/10%training.ipynb` (scratch cell) | `--arch base --fraction 0.1 --init scratch` |
| `swinv2large/30%training.ipynb` (mae cell) | `--arch large --fraction 0.3 --init foundation` |
| `nnunetv2/100%training-mae.ipynb` | `--arch nnunet --fraction 1.0 --init foundation` |

---

## 3. Bugs fixed

### 3.1 The phantom optimizer (correctness — silent training failure)

Present in the **scratch-variant cells** of `swinv2small`, `swinv2base` and
`swinv2large`. The original code did this:

```python
model     = build_model(..., use_foundation=False)   # model A
optimizer = optim.AdamW(model.parameters(), ...)     # optimizer bound to A
...
else:
    if use_foundation:
        model = build_model(..., use_foundation=True)  # model REBOUND to B
```

The training loop then forwarded through **model B** while `optimizer.step()`
updated **model A**'s parameters. Calling one of those cells with
`use_foundation=True` produced a model that trained on nothing at all — loss
would move (gradients existed) but the evaluated weights never updated.

The author found this and fixed it in the `-mae` cells, whose comment reads
*"[Core Fix]: Construct the correct model directly here in one step; do not
overwrite it later"* — but never back-ported the fix to the scratch cells.

**Fixed:** `build_segmentation_model()` returns one fully-initialised model, and
the optimizer is created from it afterwards. Structurally impossible to
reintroduce.

`nnunetv2`'s scratch cell did not have this bug.

### 3.2 `SwinUNETR(img_size=...)` crashes on current MONAI

MONAI removed the `img_size` constructor argument (it now infers spatial size at
runtime). The notebooks pass it unconditionally, so they raise `TypeError` on
MONAI ≥ 1.5.

**Fixed:** `build_swin_unetr()` inspects the constructor signature and only
passes `img_size` if that version accepts it. Works on both old and new MONAI
rather than pinning a version.

### 3.3 Non-deterministic "seeded" split

The split calls `random.seed(42)` and looks reproducible, but it iterates
`Path.rglob()` directly, whose order is filesystem-dependent. The same dataset
therefore produced **different train/test partitions on different machines**,
silently undermining any scratch-vs-foundation comparison run across machines.

**Fixed:** `build_subject_index()` sorts its results, so the seeds now do what
they appear to do.

⚠️ **This changes which patients land in train vs test** compared to whatever
partition a given machine happened to produce before. Any checkpoint trained
before this change was evaluated on a different split. Re-run both arms of a
comparison after this change; do not compare an old run against a new one.

The partition *logic* is otherwise byte-identical — verified against a
transcription of the original code at 10%, 30% and 100% (identical file lists).

### 3.4 Hardcoded, inconsistent GPU pinning

Files variously set `os.environ["CUDA_VISIBLE_DEVICES"]` to `"0"`, `"1"` or
`"2"` with no pattern, and unconditionally used CUDA autocast — crashing on any
machine without CUDA.

**Fixed:** `--device auto|cpu|cuda|cuda:N|mps`. AMP and `GradScaler` follow the
selected device instead of assuming CUDA.

### 3.5 Hardcoded dataset paths

`swinv2large` had a real lab path baked in (`/data17/user/mx79/...`); the others
had `/path/to/your/dataset/...` placeholders that would fail on any machine.

**Fixed:** paths come from `petct/config.py`, overridable by environment
variable (`PETCT_AUTOPET_ROOT`, `PETCT_PRETRAIN_ROOT`, `PETCT_WEIGHTS_DIR`,
`PETCT_OUTPUT_DIR`) or a `--data-root` flag.

---

## 4. Robustness added

These are new safety nets, not behaviour changes.

- **Weight-transfer verification.** `load_foundation_weights()` fails loudly if
  the checkpoint's key prefix doesn't match. Previously `strict=False` would
  silently transfer *nothing* on a prefix mismatch, leaving you convinced you
  were fine-tuning a pretrained model while actually training from scratch.
- **MPS memory management.** MPS's allocator doesn't reliably free memory
  between iterations for these 3D models; without explicit cache clearing it
  creeps past the watermark and OOMs partway through an epoch (observed on a
  16 GB machine). `devices.empty_cache()` handles CUDA and MPS.
- **`--sw-batch-size`.** Sliding-window inference batch size is now a flag. The
  hardcoded `4` OOMs on full-size validation volumes on smaller GPUs.
- **Empty-split guard.** `--fraction` small enough to select zero scans now
  errors instead of silently entering a training loop over nothing.
- **Final-epoch validation.** Evaluation runs on the last epoch even when
  `epochs % val_interval != 0`. Previously a 100-epoch run with
  `val_interval=20` happened to align; a 30-epoch run would have finished
  without ever evaluating.
- **`--seed`** seeds Python/NumPy/torch for reproducible runs.

---

## 5. Deliberately unchanged

Preserved exactly, including where it is arguably suboptimal, because changing
it would alter results:

- Split arithmetic: 200 test patients, 400-scan multi-scan cap, 700-scan pool,
  seeds 42 and 1024.
- All preprocessing constants: 3×2×2 mm spacing, −500 HU body threshold, 3-voxel
  crop padding, per-modality z-scoring.
- MAE hyperparameters: 12×16×16 mask blocks, 0.5 mask ratio, zero imputation,
  1.0/0.2 masked/visible loss weighting.
- Per-architecture learning rates (1e-4 transformers, 5e-4 nnU-Net), AdamW,
  weight decay 1e-5, cosine annealing to 1e-6, gradient clipping at 1.0.
- `DiceCELoss(to_onehot_y=True, softmax=True)`; Dice and HD95 with
  `include_background=False`.
- Sliding-window ROI 96×128×128 at 0.5 overlap.
- Checkpoint naming: `latest_seg_<tag>.pth` / `best_seg_<tag>.pth`, and
  resume-from-latest behaviour.
- nnU-Net weight transfer still skips `seg_outputs` (the reconstruction head is
  not a classification head), while SwinUNETR transfers everything.

---

## 6. Known limitation carried over

`scripts/pretrain.py --arch nnunet` and `--arch nnunet` fine-tuning use
`PlainConvUNet` from `dynamic_network_architectures` with **hand-written stage
parameters** replicating nnU-Net v2's `3d_fullres` heuristics for a 96×128×128
input.

That is the nnU-Net *network*, but **not** the nnU-Net *pipeline*. Real nnU-Net
fingerprints your dataset and plans patch size, spacing, stage count and batch
size itself — that auto-configuration is the main thing that makes nnU-Net
strong, and it is bypassed here.

This was appropriate for the original purpose (a like-for-like CNN comparison
against SwinUNETR at a fixed patch size). If a genuine nnU-Net baseline is
needed, install `nnunetv2` from PyPI and use its own CLI
(`nnUNetv2_plan_and_preprocess` → `nnUNetv2_train`) rather than this code path.

---

## 7. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PETCT_AUTOPET_ROOT` | `./sample_data` | labelled AutoPET data |
| `PETCT_PRETRAIN_ROOT` | `./PETCTfoundation` | unlabelled `.npy` corpus |
| `PETCT_WEIGHTS_DIR` | `./weights` | pretrained checkpoints |
| `PETCT_OUTPUT_DIR` | `./runs` | fine-tuning outputs |
