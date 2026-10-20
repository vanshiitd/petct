# nnU-Net v2 fine-tuned from MAE-pretrained weights

Branch `nnunet-mae-finetune` (from `data-fix-nnunet`).
Dataset503_AutoPET_MAEprep, frozen split `splits/autopet_v1.json`, fold 0, 500 epochs.
Machine: RTX 5000 Ada (32 GB), nnunetv2 2.8.1, torch 2.14.0+cu130.

---

## Headline

| metric (200 held-out test patients) | Dataset501, from scratch | **Dataset503, MAE fine-tuned** | change |
|---|---|---|---|
| **Dice, tumour-positive cases** | 0.1755 | **0.2962** | **+69%** |
| Dice, median | 0.1239 | **0.2718** | +119% |
| complete misses (empty prediction) | 0 of 110 | **0 of 110** | — |
| HD95 mean, mm | 234.5 | **229.2** | −2% |
| HD95 median, mm | 190.3 | **121.5** | **−36%** |
| HD95 coverage | 110 of 110 | 110 of 110 | — |
| **tumour-free cases with any false positive** | 90 of 90 (100%) | **81 of 90 (90%)** | −9 cases |
| **mean FP volume** | 151.3 mL | **38.9 mL** | **−74%** |
| mean FP volume, tumour-free cases | 127.2 mL | **31.8 mL** | **−75%** |
| mean FN volume | 15.7 mL | 29.9 mL | +90% (worse) |

Validation (76 cases) moves the same way, so this is not a test-set artefact:

| metric (76 val) | Dataset501 | Dataset503 |
|---|---|---|
| Dice, tumour-positive | 0.1396 | **0.2719** |
| HD95 median, mm | 144.1 | **83.6** |
| tumour-free with any FP | 38 of 38 (100%) | **31 of 38 (82%)** |
| mean FP volume | 150.1 mL | **30.8 mL** |
| mean FN volume | 26.0 mL | 28.7 mL |

Both runs are scored by the same script against the **same original `tumorSeg.nii.gz`**, after
Dataset503's predictions are mapped out of its cropped grid — so the numbers are comparable.

## What this does and does not show

The false-positive problem that defined the from-scratch baseline is largely gone: a quarter of
the spurious volume, and nine tumour-free patients now come back completely clean where none did
before. Dice roughly doubles on the median case. The trade is real but small in absolute terms:
false-negative volume rises from 15.7 to 29.9 mL, i.e. the model is less willing to call
something a lesion, which costs some true disease.

**This is not a clean measurement of pretraining.** Dataset503 changes three things at once
against Dataset501:

1. **initialisation** — MAE-pretrained encoder and decoder instead of random
2. **input preparation** — body-cropped, PET as channel 0, and already z-scored per channel, so
   nnU-Net's per-image z-score never runs
3. **patch size** — 192x96x128 on cropped volumes instead of 128x128x128 on full ones

Point 2 alone addresses the defect diagnosed in `nnunet_baseline.md`: per-image z-scoring makes
the same tissue land anywhere from below the image mean to 26 standard deviations above it, and
a pooled-threshold test put the loss at ROC AUC 0.825 versus 0.879 for a global scale. A large
part of this improvement is very likely that fix rather than the pretrained weights.

**The experiment that separates them is a from-scratch run on Dataset503** — same data, same
patch size, same everything, random initialisation. That is deferred, so for now the honest
claim is: *this configuration, taken as a whole, is much better than default nnU-Net on raw
PET*, and the share attributable to pretraining is unmeasured.

---

## Step 1 — how pretraining prepared the input

`scripts/preprocess_autopet.py` (pretraining corpus) and `petct/transforms.py`
(`AutoPETPreprocessd`) were run on the same scans and compared voxel-wise: **max absolute
difference exactly 0.0** over four scans. They do not differ.

| property | value |
|---|---|
| channel order | **channel 0 = PET, channel 1 = CT** (Dataset501 used the opposite) |
| target spacing | (3.0, 2.0, 2.0) mm, z-y-x |
| source spacing rounded first | **yes** — `np.round(pet_img.GetSpacing())` |
| effect on this data | 2.0364 rounds to 2.0, so zoom = (1, 1, 1): **no resampling happens at all** |
| body crop | CT > −500 HU, largest connected component, 3-voxel pad, clipped to the PET shape |
| normalisation | per-channel z-score, computed **over the cropped volume** |

Because the resampling is the identity, the prepared volume is a plain crop of the original
voxel grid, and mapping a prediction back is exact rather than approximate.

## Step 2 — Dataset503 export and its inverse

`scripts/export_nnunet_maeprep.py` runs `AutoPETPreprocessd` itself rather than a copy, writes
PET as `_0000` and CT as `_0001`, declares both channels `noNorm` so nnU-Net does not normalise
already-z-scored data, and saves `inverse/<case>.json` per case.
`scripts/map_back_predictions.py` consumes that to undo the crop.

Exported: **1014 cases, 0 refused, 42.4 GB.** Prepared shapes min 200x122x180, median
325x188x218, max 661x236x252.

Sanity checks on the export: channel 0 correlates **1.000000** with the PET crop and channel 1
with the CT crop; both channels have mean 0 and std 1; spacing is kept at the true 2.0364 mm and
the origin lands exactly on `TransformIndexToPhysicalPoint` at the crop corner.

### Acceptance check

Prepared labels pushed back through `map_back_predictions.py` and compared with the original
`tumorSeg.nii.gz`, over 200 test and 20 train cases:

| | |
|---|---|
| Dice exactly 1.0 | **217 of 220** |
| mean | 0.999908 |
| min | 0.990037 |

The three below 1.0 are **not** a mapping error: their lesions extend past the body crop box, so
those voxels were never in the prepared volume. Measured across all 501 tumour-positive scans,
the crop clips **18 of them (3.6%)**, losing **0.112%** of lesion voxels overall and at worst 16%
of a single scan — a ceiling on pooled Dice of about **0.9989**, inherited from the pretraining
preparation rather than introduced here.

## Step 3 — plans that match the checkpoint

The default planner would have made the checkpoint unloadable. For this dataset's anisotropic
voxels it chose a final stride of **`[2, 1, 2]`**, so the deepest stage downsamples only two axes.

`nnUNetPlans_MAE.json` keeps every data decision and replaces only the architecture:

| | default plan | nnUNetPlans_MAE |
|---|---|---|
| spacing | [3.0, 2.0364, 2.0364] | unchanged (so the prepared data is not resampled again) |
| patch size | [192, 96, 128] | unchanged — **already divisible by 32**, no rounding needed |
| batch size | 2 | unchanged |
| normalisation | NoNormalization x2 | unchanged |
| kernels | all [3,3,3] | all [3,3,3] |
| **strides** | **[1,1,1] then [2,2,2]x4, [2,1,2]** | **[1,1,1] then [2,2,2]x5** |
| stages / features | 6 / 32,64,128,256,320,320 | same |

Preprocessed: 814 cases, **52.4 GB** (not the 15–20 GB first estimated — z-scored float32
compresses far worse than the original volumes, which were largely constant air).

Note: the MAE plans kept `data_identifier: nnUNetPlans_3d_fullres`, so the preprocessed tensors
live in that folder. Harmless here, since only patch size and architecture changed and neither
affects preprocessing — but a future variant that changes spacing or normalisation must take its
own identifier or it will silently collide.

## Step 4 — the checkpoint loads, proven twice

`scripts/convert_mae_to_nnunet.py` strips the `module.`/`nnunet.` prefixes, drops the 10
`decoder.seg_layers` tensors (in pretraining they reconstructed two image channels; in
fine-tuning the same shapes predict two classes) and repackages as `{"network_weights": ...}`.
282 of 292 tensors kept, 88.62M parameters.

1. **Built from the plans and matched against the checkpoint:** 282 non-seg tensors, **0
   missing, 0 unexpected, 0 shape mismatches**, first conv `(32, 2, 3, 3, 3)`.
2. **Through nnU-Net's own `load_pretrained_weights`:** all **282/282 tensors bit-identical** to
   the checkpoint afterwards, both probed encoder layers (stage 0 and stage 5) differing from
   random initialisation, only the 10 seg-layer tensors left fresh.

This mattered because nnU-Net loads pretrained weights leniently: a mismatched architecture
would have transferred nothing while still logging that it had loaded a file.

## Step 5 — training

500 epochs needed no new code — nnU-Net ships `nnUNetTrainer_500epochs`, so the installed
package stays stock. (A 300-epoch trainer was written and then removed when the budget changed.)

```
nnUNetv2_train 503 3d_fullres 0 -p nnUNetPlans_MAE -tr nnUNetTrainer_500epochs \
    -pretrained_weights weights/nnunet_mae_pretrained_for_nnunet.pth
```

Confirmed from nnU-Net's own log: `This split has 738 training and 76 validation cases.` and the
pretrained-weights banner.

| | |
|---|---|
| epoch time | 58.4 s (the pilot's 65.8 s included warm-up) |
| total | **7.8 h**, 500/500 epochs, zero errors |
| GPU | 98–99% throughout, 8.7 GB of 32.8 GB, 81–83 °C |
| final EMA pseudo-Dice | **0.3536** (Dataset501 reached 0.3633 in 1000 epochs) |

## Step 6 — prediction and scoring

`checkpoint_final.pth` (fixed in advance, not `checkpoint_best`), fold 0. 200 test cases at
**8.0 s/case** — five times faster than Dataset501's 43 s/case, because the volumes are
body-cropped. All 200 test and 76 val predictions mapped back with **0 failures**, then scored
against the original labels.

---

## Files

| | |
|---|---|
| raw dataset | `D:\data\nnunet\raw\Dataset503_AutoPET_MAEprep` (42.4 GB) |
| preprocessed | `C:\nnunet_preprocessed\Dataset503_AutoPET_MAEprep` (52.4 GB) |
| model | `D:\data\nnunet\results\Dataset503_AutoPET_MAEprep\nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres\fold_0` |
| converted weights | `D:\data\petct\weights\nnunet_mae_pretrained_for_nnunet.pth` |
| predictions | `…\predictions\test_503_mapped`, `…\predictions\val_503_mapped` |
| per-case scores | `reports\eval_503_test.csv`, `reports\eval_503_val.csv` |
| training log | `reports\nnunet_mae_ft_fold0.log` |

Disk after everything: C: 270 GB free, D: 71 GB free.

Dataset501 and Dataset502 are untouched. Dataset502 (SUV + global normalisation) remains paused.
