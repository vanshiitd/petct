# SUV with global PET scaling, and false-positive clean-up

Branch `suv-and-postproc` (from `nnunet-mae-finetune`).
Dataset501/502/503 untouched. All numbers scored by `scripts/evaluate_predictions.py` against the
**original** `tumorSeg.nii.gz`, on the frozen split `splits/autopet_v1.json`.

---

## Why Dataset503 ran 500 epochs, not the 300 originally asked for

Because the instruction changed mid-task. While Step 5 was being set up, the message "continue
with 500" arrived. At that point `nnunet_ext/nnUNetTrainer_300epochs.py` had already been
written, copied into the installed package and confirmed findable by nnU-Net's class lookup --
the 300-epoch path was working. It was then deleted in favour of the stock
`nnUNetTrainer_500epochs`. The remark at the time that "nnU-Net already ships a 500-epoch
trainer" described a convenience, not the reason. Dataset504 uses 500 epochs so the two runs are
directly comparable.

---

## Part 1 — Dataset504: one global PET scale

503 prepares PET exactly as pretraining did, which includes a **per-scan z-score**. That is the
defect the earlier normalisation test measured: the same tissue lands at a different value in
every patient, so no fixed decision boundary transfers (pooled ROC AUC 0.825 per-scan against
0.879 for one global scale). 504 changes that one thing and nothing else.

### The transform

| | |
|---|---|
| channel 0 | `SUV.nii.gz`, cropped identically to 503, then **(SUV − μ) / σ** |
| **μ** | **0.270616** |
| **σ** | **0.850270** |
| fitted on | **training cases only** (738 cases, 10,642,456,124 voxels inside the body crops) |
| channel 1 | `CT_resample.nii.gz`, per-scan z-score — unchanged from 503 |
| clipping | none |

Val and test contribute nothing to μ/σ, or the normalisation would carry information out of the
held-out sets. Statistics were accumulated as running sums, so the 10.6 billion voxels never had
to be held at once. Saved to `pet_normalisation.json` beside the dataset.

### Verified before training

Eight cases checked against 503, covering single- and multi-study patients:

| check | result |
|---|---|
| geometry (size, spacing, origin, direction) | identical to 503 |
| CT channel | byte-identical to 503 |
| labels | byte-identical to 503 |
| channel 0 regressed back onto SUV | recovers **μ̂ 0.270616, σ̂ 0.850270 on every case** (spread 8e-10 and 2e-08) |
| residual against `(SUV − μ)/σ` | ≤ 6e-6 (float32 storage) |

So the PET channel really is one linear transform shared by every case, not a per-scan
normalisation in disguise.

Export: **1014 cases, 0 refused, 42.4 GB**; prepared shapes identical to 503 (min 200×122×180,
median 325×188×218, max 661×236×252). Plans, patch size (192×96×128), batch size, spacing and
splits (738/76) all as 503; the planner's anisotropic final stride `[2,1,2]` replaced with
`[2,2,2]` as before. Preprocessed 814/814, 52 GB.

Pretrained weights verified against 504's plans: **282 non-seg tensors, 0 missing, 0 unexpected,
0 shape mismatches**, only the 10 seg-layer tensors left fresh.

_Training in progress; results below once it finishes._

---

## Part 2 — false-positive clean-up

### Which organs may be masked at all

Measured on **150 tumour-positive training scans** (never val or test), as the share of real
lesion volume falling inside each organ:

| organ | share of lesion volume | scans affected | worst single scan | verdict |
|---|---|---|---|---|
| liver | **5.0761%** | 37 of 150 | 62.2% | **NOT SAFE** |
| heart | **2.9040%** | **61 of 150** | 21.1% | **NOT SAFE** |
| spleen | **2.4192%** | 21 of 150 | 70.3% | **NOT SAFE** |
| urinary_bladder | 0.2155% | 5 of 150 | 33.8% | safe |
| kidney_right | 0.1751% | 9 of 150 | 19.9% | safe |
| kidney_left | 0.0385% | 9 of 150 | 3.3% | safe |
| brain | **0.0000%** | **0 of 150** | 0.0% | safe |

**This check changed the plan.** The heart is the organ the from-scratch baseline visibly fired
on, and masking it would have looked like the single biggest win available — but it holds 2.9%
of real lesion volume across **61 of 150 scans**, so masking it would have deleted true disease
in 40% of patients. Liver and spleen are worse still. Had these been assumed rather than
measured, the result would have looked like an improvement in Dice while quietly removing real
lesions.

Allowed for masking: **brain, urinary_bladder, kidney_left, kidney_right**.

On the brain specifically: 0.0000% is zero lesion volume in 150 scans, not proof that brain
metastases never occur in this collection — it bounds the rate rather than excluding it.

### Tuning, on validation only

Both thresholds were swept on 503's validation predictions and the chosen pair applied once to
test.

| min vol (mL) | organ overlap | Dice (pos) | FP mL | FN mL | tumour-free with FP |
|---|---|---|---|---|---|
| 0.0 | off | 0.2719 | 30.80 | 28.67 | 31 of 38 |
| 0.0 | 0.25 | 0.2750 | 30.12 | 28.67 | 31 |
| **0.2** | **0.25** | **0.2755** | **29.84** | 28.81 | 29 |
| 0.5 | 0.25 | 0.2693 | 29.24 | 29.12 | 24 |
| 1.0 | 0.25 | 0.2620 | 28.31 | 31.15 | 21 |
| 2.0 | 0.25 | 0.2575 | 26.87 | 31.55 | 15 |

**Chosen: min volume 0.2 mL, organ overlap 0.25.**

The trade-off is worth stating plainly. By mean Dice the optimum is a near-flat 0.2755 against
0.2719 — a gain inside the noise. Pushing the size threshold up cleans tumour-free scans
sharply (31 → 15 with any false positive at 2 mL) but costs Dice steadily. The reason both
rules do so little is the organ result above: **the false positives are mostly large blobs on
heart, mediastinum and brown fat, which cannot be masked without deleting real disease.** Small
speckle is not the problem — on test, post-processing removed 638 small components and only 23
organ-overlapping ones, keeping 99.6% of predicted voxels.

---

## Results

### Test set — 200 patients, 110 tumour-positive, 90 tumour-free

| run | Dice mean | Dice median | complete misses | HD95 mean mm | HD95 median mm | tumour-free with FP | FP mL (all) | FP mL (tumour-free) | FN mL |
|---|---|---|---|---|---|---|---|---|---|
| 501 from scratch | 0.1755 | 0.1239 | 0 | 234.5 | 190.3 | 90 of 90 | 151.3 | 127.2 | 15.7 |
| 503 MAE fine-tuned | 0.2962 | 0.2718 | 0 | 229.2 | 121.5 | 81 of 90 | 38.9 | 31.8 | 29.9 |
| **503 + post-processing** | **0.2971** | 0.2718 | 0 | 230.4 | 121.2 | **72 of 90** | **38.2** | **31.2** | 30.1 |
| 504 SUV + global scale | 0.2882 | 0.2767 | 3 | 197.3 | 111.5 | 80 of 90 | 51.0 | 42.6 | **21.4** |
| 504 + post-processing | 0.2883 | 0.2767 | 3 | 194.3 | 110.4 | 72 of 90 | 50.7 | 42.2 | 21.5 |

**504 does not beat 503.** Dice 0.2882 against 0.2962, and false-positive volume is markedly
worse (51.0 against 38.9 mL). It does trade in a consistent direction: false-negative volume
drops to 21.4 mL from 29.9, and the median Dice is slightly higher (0.2767 against 0.2718), so
the SUV model is more willing to call something a lesion — it finds more true disease and more
spurious tissue. It also leaves three tumour-positive cases empty, where 503 left none.

Validation agreed (0.2612 against 0.2719), so this is not a test-set accident.

Post-processing leaves Dice essentially unchanged (+0.0009) but brings **nine more tumour-free
patients back completely clean** (81 → 72 with any false positive). That is a better description
of its value than the Dice column: it removes spurious findings on patients who have no disease,
which is exactly where a false positive costs most.

### Validation — 76 patients, 38 tumour-positive, 38 tumour-free

| run | Dice mean | Dice median | complete misses | HD95 mean mm | HD95 median mm | HD95 coverage | tumour-free with FP | FP mL (all) | FP mL (tumour-free) | FN mL |
|---|---|---|---|---|---|---|---|---|---|---|
| 501 from scratch | 0.1396 | 0.0699 | 0 | 211.2 | 144.1 | 38 of 38 | 38 of 38 | 150.1 | 127.2 | 26.0 |
| 503 MAE fine-tuned | 0.2719 | 0.2269 | 0 | 180.0 | 83.6 | 38 of 38 | 31 of 38 | 30.8 | 23.3 | 28.7 |
| **503 + post-processing** | **0.2755** | 0.2263 | **1** | **143.5** | **80.0** | 37 of 38 | **29 of 38** | **29.8** | **22.9** | 28.8 |
| 504 SUV + global scale | 0.2612 | 0.2073 | 1 | 151.7 | 81.7 | 37 of 38 | 32 of 38 | 33.0 | 22.6 | 29.6 |
| 504 + post-processing | 0.2614 | 0.2072 | 2 | 149.1 | 81.6 | 36 of 38 | 28 of 38 | 32.9 | 22.4 | 29.6 |

Post-processing creates **one complete miss** on validation: a tumour-positive case where every
predicted component was either below 0.2 mL or inside an allowed organ, leaving the prediction
empty. That case drops out of the HD95 mean (37 of 38 covered), which is part of why HD95 falls
so much here — the comparison is not quite like for like, and the complete-miss column is the
honest place to look.

---

## Three checks before any more GPU training

### 1. Normalisation test, with the row that was missing

The earlier table compared nnU-Net's **whole-image** z-score against global scales. It never
tested what 503 actually does: a per-scan z-score computed **inside the body crop**, where the
air that dominates a whole-image mean and std has already been removed. Same pooled test, 100
training cases, body voxels, 2.73 M sampled:

| scheme | pooled ROC AUC | best-threshold Dice | → trained model, test Dice |
|---|---|---|---|
| (a) per-image z-score, whole volume — **501** | 0.8199 | 0.1435 | **0.1755** |
| (b) per-scan z-score, inside crop — **503** | **0.8265** | 0.1415 | **0.2962** |
| (c) raw Bq/mL, one global scale | 0.8758 | 0.1748 | — |
| (d) SUVbw, one global scale — **504** | **0.8761** | 0.1681 | **0.2882** |

**This invalidates the normalisation diagnosis in `nnunet_baseline.md`.** 503's scheme scores
almost the same as 501's on this test (0.8265 against 0.8199), yet the trained models differ by
0.12 Dice. And 504's scheme is the *best* of the four by this test while training to a slightly
*worse* model than 503. The pooled-threshold test does not predict trained-model behaviour:
a threshold has nothing but absolute calibration to work with, whereas a U-Net sees spatial
context and adapts to each image's statistics.

So the 501 → 503 improvement was **not** mainly the normalisation. The remaining candidates are
the other three things 503 changed at once: MAE pretraining, body cropping, and the channel
order plus patch size. Separating them needs a from-scratch run on 503's preparation.

### 2. Fit check — the model cannot fit its own training data

50 training cases predicted with 503's `checkpoint_final`, mapped back and scored exactly as
val and test were:

| set | cases | tumour-positive | **Dice** | median | FP mL | FN mL | complete misses |
|---|---|---|---|---|---|---|---|
| **train** | 50 | 18 | **0.2950** | 0.2146 | 28.1 | 15.1 | 0 |
| val | 76 | 38 | 0.2719 | 0.2269 | 30.8 | 28.7 | 0 |
| test | 200 | 110 | **0.2962** | 0.2718 | 38.9 | 29.9 | 0 |

**Training Dice equals test Dice** (0.2950 against 0.2962). There is no generalisation gap at
all — after 500 epochs on data it has seen hundreds of times, the model scores no better than on
data it has never seen.

This is underfitting, not overfitting, and it reframes everything above. The ceiling is not
regularisation, more data, or the train/test split; the model is simply not learning the labels
it is being shown. Tuning post-processing or normalisation moves numbers by hundredths while the
fit itself is the limit.

Taken with the other two checks, the eliminations are now:

* **not the labels** — the DICOM-SEG round-trips at Dice 1.0000 (check 3)
* **not the data pairing** — audited clean, 0 mismatches over 1014 scans
* **not the split** — frozen, patient-level, no leakage, and train/val/test all agree
* **not overfitting** — zero train-test gap
* **not the PET normalisation** — the scheme that scores best on the threshold test (504) trains
  to a slightly worse model than the one that scores nearly worst (503)

What remains untested is the training configuration itself against what published AutoPET
nnU-Net work does, where roughly 0.65 Dice is reported on this dataset. The differences worth
testing, cheapest first:

1. **The CT channel is never normalised with dataset-wide statistics.** 503 and 504 declare both
   channels `noNorm`, so CT keeps only a per-scan z-score. Standard nnU-Net applies
   `CTNormalization` to CT — a global percentile clip and dataset-wide z-score. CT is the channel
   that distinguishes physiological uptake from disease anatomically, and it is the one left
   uncalibrated across patients.
2. **Patch size and batch size.** 192x96x128 at batch 2 on body-cropped volumes, against the
   128^3 at batch 2 nnU-Net planned for the uncropped data.
3. **A from-scratch run on 503's preparation**, which would finally separate the contribution of
   the MAE weights from the preparation changes that came with them.

### 3. Mask conversion — is our `tumorSeg.nii.gz` right?

Our converter reads the DICOM-SEG with SimpleITK and, when the sizes match, copies the PET's
geometry onto it rather than resampling. That is only valid if the SEG really is on the PET
grid; otherwise every model has been learning displaced labels. Rebuilt independently with
pydicom, placing **each frame by its own `ImagePositionPatient`** through the PET's geometry,
over 30 tumour-positive scans:

| | |
|---|---|
| Dice against our `tumorSeg.nii.gz` | **1.0000 on all 30** (min, median, mean all 1.0000) |
| voxel counts | identical on every scan |
| constant z shift needed | **none** — no offset, no flip |
| frames | **full, never sparse** — frame count equals PET slice count on all 30 (284 to 661) |
| SEG references | **PET series on all 30** |
| segments | a single segment, number 1 |

The conversion is correct, and the shortcut it relies on is justified: these SEGs are drawn on
the PET series, with one frame per PET slice, so the PET's geometry is the SEG's geometry.
**The labels are not the problem.**

`pydicom-seg` could not be used — version 0.4.1 imports `pydicom._storage_sopclass_uids`, which
pydicom 3.x removed, and installing it silently downgraded numpy to 1.26.4 in the `petct` venv
(breaking `np.trapezoid`). Both were reverted and the reader written directly against pydicom,
which also gives exact control over frame placement.

No comparison against an official AutoPET NIfTI release was made: it needs a registered download,
and the independent re-read above already establishes the conversion is exact, so an external
copy would add little.

## Comparison against the organisers' own code

Both repositories verified rather than assumed, and cloned:

* evaluation — **`lab-midas/autoPET`**, `val_script.py`
* conversion — **`lab-midas/TCIA_processing`**, `tcia_dicom_to_nifti.py`

### Differences visible by reading their code

**Evaluation (`val_script.py`).** Their metric set is **Dice, false-positive volume,
false-negative volume — and nothing else.** There is no HD95 at all, so every HD95 number in
these reports is ours, not a challenge metric.

| | organisers | ours |
|---|---|---|
| connected components | `cc3d`, **connectivity 18** | `scipy.ndimage.label`, default **connectivity 6** |
| Dice when both masks are empty | `2*0/0` → NaN | explicitly NaN, excluded |
| Dice on a tumour-free case with a false positive | `2*0/sum` → 0 | 0, and we report it separately |
| HD95 | not computed | computed, in mm |
| aggregation | not in the repo | mean over tumour-positive cases |

Connectivity 18 merges components that touch along an edge, which 6-connectivity leaves
separate. That makes their component count smaller and so their false-positive volume
*different*, not simply larger or smaller.

**Conversion (`tcia_dicom_to_nifti.py`).** Four concrete differences from ours:

1. **SUV decay reference.** Theirs uses `AcquisitionTime`; ours uses `SeriesTime`, falling back
   to `AcquisitionTime`. On `PETCT_0011f3deaf` those differ by 12 min 41 s, which is a 7.7%
   difference in the decayed dose and so in the SUV factor. This affects Dataset504 but not 503,
   which z-scores raw PET per scan.
2. **CT resampling.** Theirs is `nilearn.image.resample_to_img(..., fill_value=-1024)`, whose
   default interpolation is continuous (cubic); ours is `sitk.Resample` with **linear**
   interpolation and fill **−1000**.
3. **Mask handling.** Theirs transposes `(2,1,0)` and conditionally flips on
   `ImageOrientationPatient[4] == 1`, then borrows the PET affine. Ours reads the SEG with
   SimpleITK and copies the PET geometry. Our independent per-frame re-read already showed ours
   is exact (Dice 1.0000 on 30 scans), so this is about whether theirs agrees.
4. **Orientation.** They convert with `dicom2nifti(reorient=True)`, which can place the array in
   a different axis order from ours. Comparisons must therefore go through physical space, not
   raw array indices.

**And from their README, a difference in the baseline itself:** the official nnU-Net baseline was
trained with **initial learning rate 1e-4**, against the nnU-Net default of 1e-2 that 501, 503
and 504 all used. The same README confirms the baseline used **SUV PET and resampled CT** as its
two channels, 1000 epochs, 3d_fullres, 5-fold cross validation.

### 1. Metric parity — measured

The organisers' `val_script.py`, imported unmodified and run over 503's 200 test predictions
against our labels:

| metric | organisers | ours | difference |
|---|---|---|---|
| **Dice, tumour-positive cases** | **0.2962** | **0.2962** | **exact** |
| Dice, median over positives | **0.2718** | **0.2718** | **exact** |
| FP volume, mL (all cases) | 38.8661 | 38.8898 | 0.024 (0.06%) |
| FN volume, mL (all cases) | 29.7713 | 29.8892 | 0.118 (0.4%) |
| tumour-free with any false positive | 81 of 90 | 81 of 90 | same |

**The metric is not the explanation.** Dice agrees exactly; the sub-percent volume gap is the
connectivity difference (18 against 6), which changes how adjacent blobs are grouped and
therefore which ones count as wholly false.

Two things their aggregation makes visible that ours did not:

* **Dice averaged over ALL 200 cases is 0.1706**, not 0.2962, because a tumour-free case with
  any false positive scores a hard zero and there are 81 of them. Published autoPET figures are
  over lesion-positive cases, so 0.2962 remains the number comparable with ~0.65 — but 0.1706 is
  what a leaderboard averaging over everything would show.
* **9 cases are NaN** (`2*0/0`): tumour-free patients the model left empty. Their script does not
  special-case this, so an aggregate that ignores NaN silently drops the model's nine cleanest
  results.

### 2. Conversion diff — **a severe label bug in our pipeline**

Their conversion run unmodified on 20 patients (10 tumour-positive, 10 tumour-free, 4
multi-study), compared against ours through physical space.

**8 of the 10 tumour-positive cases have a different mask:**

| case | SEG Dice, theirs vs ours |
|---|---|
| `PETCT_3b1c9155f5` | 0.0149 |
| `PETCT_f144b214af` | 0.0178 |
| `PETCT_8a52353a72` | 0.0705 |
| `PETCT_9afd440677` | 0.1038 |
| `PETCT_223f70f0b0` | 0.1369 |
| `PETCT_b327726c24` | 0.3516 |
| `PETCT_17d334cb6c` | 0.4801 |
| `PETCT_510fb36781` | 0.4891 |
| `PETCT_af73b85f2c`, `PETCT_b53ba7c6bf` | 1.0000 |

The voxel counts are **identical** in every case, so this is the same mask placed differently,
not a different segmentation.

**Which is right? The PET decides.** A lesion label must sit on elevated uptake:

| case | our mask, uptake vs body | their mask |
|---|---|---|
| `PETCT_3b1c9155f5` | **0.68x** — dimmer than background | **3.29x** |
| `PETCT_510fb36781` | 4.06x | **9.15x** |
| `PETCT_af73b85f2c` | 16.76x | 16.76x (identical) |

**Ours is wrong.** And the fix is exact — flipping our mask along y reproduces their output bit
for bit:

| case | as stored | **flipped in y** | flipped in x |
|---|---|---|---|
| `PETCT_3b1c9155f5` | 0.0149 | **1.0000** | 0.0014 |
| `PETCT_f144b214af` | 0.0178 | **1.0000** | 0.0000 |
| `PETCT_510fb36781` | 0.4891 | **1.0000** | 0.4148 |
| `PETCT_af73b85f2c` | **1.0000** | 0.3644 | 0.0000 |

#### Scope: 63% of tumour-positive scans

Measured over all 501 tumour-positive scans, using PET uptake as the arbiter (is the mask on
brighter tissue as stored, or flipped?):

| | |
|---|---|
| mask is on **brighter tissue when flipped** | **308 of 501 (61.5%)** |
| mask uptake ratio below 1.5x as stored | 155 (30.9%) |
| median uptake ratio **as stored** | **2.57x** |
| median uptake ratio **flipped** | **4.30x** |

The SEG's `ImageOrientationPatient[4]` predicts it almost perfectly:

| orientation[4] | scans | flip improves uptake | reading |
|---|---|---|---|
| **−1.0** | 317 | **305 (96.2%)** | our mask is flipped |
| +1.0 | 184 | 3 (1.6%) | our mask is correct |

So roughly **317 of 501 tumour-positive scans carry a left-right mirrored lesion mask**.
Tumour-free scans are unaffected — an empty mask cannot be flipped.

#### Why this is the root cause

`dicom_to_nifti.py` reads the SEG with SimpleITK and then calls `seg_bin.CopyInformation(pet)`,
which **stamps the PET's geometry onto the mask instead of resampling it**. When the SEG's own
orientation differs from the PET's — which it does for 317 of 501 scans — that silently mirrors
the mask. The organisers' script handles exactly this case with an explicit conditional flip.

This explains the central finding of the fit check. The model was shown correct labels on about
a third of tumour-positive scans and mirrored labels on the rest; no network can fit both, which
is why **training Dice equals test Dice at 0.29** and why nothing else moved it. It also explains
the gap to the published ~0.65 on this dataset.

#### Why the earlier mask check missed it

Check 3 rebuilt the mask with an independent reader and got Dice 1.0000 on 30 scans — but that
reader placed each frame by its z position and **assumed in-plane alignment, the same assumption
the converter makes**. Two readers sharing one blind spot agree with each other and prove
nothing. `PETCT_510fb36781` is in both samples: 1.0000 against my reader, 0.4891 against the
organisers'. Only code written independently of ours could expose this, which is what comparing
against the organisers' conversion was for.

#### Other differences found (minor by comparison)

* **SUV decay reference.** Theirs decays the dose to `AcquisitionTime`, ours to `SeriesTime`.
  Their SUV runs higher by a median of about 4-5%, up to 19% on one scan
  (`PETCT_3b1c9155f5`, ratio 1.1885). This affects Dataset504, which scales SUV globally; 503 is
  immune because a per-scan z-score cancels any per-scan factor.
* **CT resampling.** Theirs uses `nilearn.resample_to_img` (cubic, fill −1024), ours SimpleITK
  linear with fill −1000. Maximum absolute differences of 400-4000 HU, concentrated at edges
  and outside the original field of view.
* **Orientation.** They convert with `dicom2nifti(reorient=True)`, so their arrays sit on a
  different grid from ours; all comparisons above went through physical space.

To run their script at all it needed one shim: it calls `pydicom.read_file`, which pydicom 3.x
removed (their requirements pin 2.3.0). `read_file` was a deprecated alias of `dcmread`, so the
name was restored rather than editing their code or downgrading the stack.

### 3. Faithful reproduction proposal — not applicable yet

Step 3 was conditional on checks 1 and 2 finding nothing meaningful. Check 2 found a label bug
affecting roughly 63% of tumour-positive scans, so proposing a reproduction now would be
premature: **every model trained so far — 501, 503, 504 — learned from mirrored labels on most
positive scans, and every number in every report rests on them.**

## Environment notes

TotalSegmentator 2.18.0 runs in its own venv (`D:\data\totalseg_venv`, torch 2.14.1+cu130) so
its pinned nnU-Net cannot disturb `nnunet_venv` (2.14.0+cu130) or the `petct` env
(2.11.0+cu128). All three verified CUDA-enabled afterwards.

Organ segmentation: 426 cases (76 val + 200 test + 150 tumour-positive train), fast mode (3 mm),
`roi_subset` limited to the seven organs above, multilabel output. **38.9 s per case** through
the Python API in one process, against 121 s per case invoking the CLI per scan — the difference
is model loading, which a single long-lived process pays once. Masks come out on
`CT_resample.nii.gz`'s grid, which is the PET grid, so they align voxel for voxel with
predictions mapped back to the original scan; label ids confirmed unchanged by `roi_subset`.

Three mistakes of mine during that run, all corrected:

1. Piping TotalSegmentator's output through `Select-Object -First 6` closed the PowerShell
   pipeline and killed the process at case 40. Relaunched with file redirection; completed
   cases were skipped.
2. The three tallest scans (576–661 slices, ~100M voxels) failed with TotalSegmentator reporting
   "insufficient CPU RAM" while 100 GB was free. The real error is a Windows pipe write limit,
   `OSError WinError 87`, on transfers that large. Worked around by segmenting in overlapping
   halves and stitching.
3. That workaround's first version lacked `if __name__ == "__main__":`, so every worker process
   TotalSegmentator spawned re-ran the whole script. Added the guard.

Disk after everything: see final section.
