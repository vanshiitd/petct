# Dataset505 — the same model on corrected labels

**Question.** Dataset503 fine-tuned our MAE-pretrained nnU-Net and reached test Dice
0.2962, against ~0.65 in the published AutoPET work. A 50-case fit check then showed
train Dice 0.2950 against test 0.2962 — the model was no better on data it had
trained on than on data it had never seen. That is not overfitting and not a
generalisation gap; it is a model that cannot fit its own labels.

The cause was in the converter: `seg_bin.CopyInformation(pet)` stamped the PET's
geometry onto the lesion mask instead of resampling the mask onto the PET's grid.
Equal size does not imply equal geometry, and for 317 of the 501 tumour-positive
scans the SEG's orientation differs from the PET's, so the mask was silently
mirrored left–right. Nothing downstream could notice: the mask kept its shape, its
voxel count and its file size.

Dataset505 re-runs Dataset503 **unchanged** on labels reconverted with the
organisers' own pipeline. Only the data root differs.

## What was held fixed

| | 503 | 505 |
|---|---|---|
| Preparation | MAE prep: 3×2×2 mm, body crop, per-scan z-score | same |
| Channels | 0 = PET, 1 = CT, both `noNorm` | same |
| Plans | `nnUNetPlans_MAE`, patch 192×96×128, batch 2 | same |
| Architecture | PlainConvUNet, 6 stages, [32,64,128,256,320,320], isotropic 3³ | same |
| Init | `nnunet_mae_pretrained_for_nnunet.pth`, 282/282 tensors matched | same |
| Trainer | `nnUNetTrainer_500epochs`, fold 0 | same |
| Split | `splits_final.json`, 738 train / 76 val | **byte-identical** (SHA-256 match) |
| Test set | 200 patients, 110 tumour-positive | same patients |

Only difference: `C:\data\autopet_nifti_v2` in place of `D:\data\autopet_nifti`.
Inherited from the organisers' script and deliberately kept: SUV decay referenced
to `AcquisitionTime` rather than `SeriesTime` (theirs runs 4–5% higher, which the
per-scan z-score removes), and CT resampled with cubic interpolation and a −1024
fill rather than linear and −1000.

## Evidence the labels moved, before any training

The nnU-Net fingerprint measures intensity inside the labels, on the prepared data
the network actually sees. Channel 0 is PET in z-score units:

| foreground (lesion) PET | 503 | 505 |
|---|---|---|
| median | 1.62 | **4.23** |
| mean | 3.55 | **5.97** |

The labels now sit on bright tracer instead of near-background tissue. Independently,
over all 501 tumour-positive scans, median lesion uptake is 6.14× the body median as
stored against 1.46× flipped; before the fix it was 2.57× stored against 4.30× flipped.

The map-back from the prepared grid to original geometry was checked to be
voxel-exact on three test cases, including an 8-voxel lesion, so the scores below
are the model's and not the plumbing's.

## Results

Scored in original scan geometry against the reconverted ground truth. Dice is over
tumour-positive cases; FP/FN volumes are over all cases.

### Test — 200 patients, 110 tumour-positive

| metric | 503 mirrored | **505 corrected** | change |
|---|---|---|---|
| Dice | 0.2962 | **0.6421** | ×2.17 |
| median Dice | 0.2718 | **0.7580** | ×2.79 |
| positives scoring Dice 0 | 20 | 4 | −16 |
|  of those, empty prediction | 0 | 1 | +1 |
| HD95 mm (mean) | 229.2 | 128.0 | −44% |
| median HD95 mm | 121.5 | 39.0 | −68% |
| FP volume mL | 38.89 | 15.94 | −59% |
| FN volume mL | 29.89 | 5.45 | −82% |
| tumour-free cases with any FP | 81/90 | 63/90 | −18 |

### Validation — 76 patients, 38 tumour-positive

| metric | 503 mirrored | **505 corrected** | change |
|---|---|---|---|
| Dice | 0.2719 | **0.6963** | ×2.56 |
| median Dice | 0.2269 | **0.7972** | ×3.51 |
| positives scoring Dice 0 | 7 | 2 | −5 |
| HD95 mm (mean) | 180.0 | 79.7 | −56% |
| median HD95 mm | 83.6 | 36.7 | −56% |
| FP volume mL | 30.80 | 10.21 | −67% |
| FN volume mL | 28.67 | 4.28 | −85% |
| tumour-free cases with any FP | 31/38 | 26/38 | −5 |

### Fit check — the same 50 training cases, 18 tumour-positive

| metric | 503 mirrored | **505 corrected** |
|---|---|---|
| train Dice | 0.2950 | **0.7461** |
| median | 0.2146 | 0.8046 |
| FN volume mL | 15.06 | 3.11 |

| | 503 | 505 |
|---|---|---|
| train − test | **−0.0012** | **+0.1040** |
| train − val | +0.0231 | +0.0498 |

This is the result the whole exercise was for. On 503 the model scored *the same* on
data it had trained on as on data it had never seen, which is what a model does when
the labels carry no learnable signal. On 505 it scores 0.10 above test and 0.05 above
validation — an ordinary, modest train/held-out gap. The network is now fitting the
labels and generalising from them.

## Training

| | 503 | 505 |
|---|---|---|
| Best EMA pseudo-Dice | 0.3536 | **0.7857** |
| Wall clock (500 epochs) | ~8.3 h | 8 h 16 m |
| Epoch time | ~58 s | ~58 s |

Early epochs, same recipe, same schedule:

| epoch | 503 pseudo-Dice | 505 pseudo-Dice |
|---|---|---|
| 0 | 0.0258 | 0.1239 |
| 1 | 0.0781 | 0.2438 |
| 2 | 0.0621 | 0.2147 |

505's second epoch already beats what 503 reached after many. 503's *final* best EMA
pseudo-Dice, 0.3536, is below where 505 sits within the first few percent of training.

## What this does and does not settle

Test Dice 0.6421 and val 0.6963 land in the published AutoPET range, so the
pipeline, the architecture and the MAE initialisation were never the limitation —
the labels were. 501's earlier failure (0.1755) is not explained by this alone,
since it also used global CT normalisation and 128³ patches; it trained on the same
mirrored labels, so its number is not interpretable either way.

Still open, and not answered here: how much of 505's score comes from the MAE
pretraining as opposed to the architecture and data. That needs the from-scratch
control on Dataset505, which is the next run.

The residual false-positive load is real and unchanged in character: 63 of 90
tumour-free test patients still carry some false positive, 19.6 mL on average. The
earlier post-processing sweep was tuned on mirrored labels and should be redone.

## Artifacts

| | |
|---|---|
| Training log | `D:\data\reports\nnunet_505_fold0.log` |
| Checkpoints | `D:\data\nnunet\results\Dataset505_AutoPET_MAEprep_v2\nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres\fold_0` |
| Per-case scores | `reports\eval_505_{test,val,train}.csv` |
| Predictions | `D:\data\nnunet\predictions\{test,val,train}_505_mapped` |
| Ground truth | `D:\data\nnunet\gt_v2_{test,val,fitcheck}` |

Reproduce the tables:

```
python scripts/compare_eval_runs.py --a-label 503 --b-label 505 \
    --pair test reports/eval_503_test.csv reports/eval_505_test.csv \
    --pair val  reports/eval_503_val.csv  reports/eval_505_val.csv \
    --pair train reports/eval_503_train.csv reports/eval_505_train.csv
```
