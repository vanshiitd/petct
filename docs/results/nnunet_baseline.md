# nnU-Net v2 baseline on AutoPET — running report

Working branch: `data-fix-nnunet` (from `main` @ 218d118)
Machine: RTX 5000 Ada (32 GB), torch 2.11.0+cu128, MONAI 1.6.0, `petct` venv at `D:\data\petct\.venv`

---

## Step 0 — code update

| | |
|---|---|
| Uncommitted change found | `petct/data.py` — the `SpatialPadd` fix from the earlier SwinUNETR run |
| Action | committed to branch `workstation-local` as **a9739f3** (not discarded) |
| `main` after pull | **218d118** "Add predict_overlays.py for prediction-vs-truth figures" |
| Working branch | `data-fix-nnunet` created from 218d118 |
| Dependencies | `pip install -r requirements.txt` — all already satisfied, nothing changed |
| Git identity | none was configured; set **repo-locally only** to vanshiitd / vanshsaini.iitd@gmail.com so commits could be made. No push credentials set up. |

Note: `pydicom` is not installed in the venv, so the audit script uses SimpleITK only.

---

## Step 1 — study-pairing audit

Script: `scripts/check_study_pairing.py` (new, committed).
Reads **one** DICOM member out of each of `CT.zip`/`PT.zip`/`SEG.zip` per patient and compares
StudyInstanceUID (0020|000D) and StudyDate (0008|0020).

Command run:

```
python scripts/check_study_pairing.py --source D:\data\dataset \
    --out D:\data\reports\pairing_audit.csv --tmp-dir D:\data\convert_tmp
```

### Result — **59 mismatched patients** (expected ~49)

| | count |
|---|---|
| patients audited | 897 |
| complete (all three archives) | 894 |
| incomplete (never fully downloaded) | 3 — `PETCT_6c660140ca`, `PETCT_a41d59682f`, `PETCT_d4f3375362` |
| **study-mismatched** | **59** |
| … of those, currently in `autopet_nifti` (i.e. used as training data) | **53** |
| … of those, already in `excluded_nifti` | 6 |

### All 59 are genuine, not audit artefacts

Every mismatched patient also has **different StudyDates** across the mixed modalities —
zero cases where the UIDs differ but the date is the same. So these are real cross-visit
mixes, not two same-day studies or a metadata quirk.

Shape of the mismatch:

| pattern | count | meaning |
|---|---|---|
| CT = PT, SEG differs | 20 | mask from a different visit than the images |
| PT = SEG, CT differs | 19 | CT from a different visit than the PET+mask |
| CT = SEG, PT differs | 17 | PET from a different visit |
| all three differ | 3 | three separate visits |

56 patients mix 2 studies, 3 patients mix 3 studies.

### Blank-CT verdict — hypothesis confirmed

All **3** blank-CT patients are in the mismatch list:

| patient | CT date | PT date | SEG date |
|---|---|---|---|
| `PETCT_234f8427c0` | 2005-12-03 | 2004-12-09 | 2004-12-09 |
| `PETCT_86153b2974` | 2005-04-03 | 2003-06-05 | 2006-10-06 |
| `PETCT_e344879c2e` | 2007-03-08 | 2006-11-17 | 2007-03-08 |

Each has its CT from a different visit than its PET. Resampling that CT onto the other
visit's PET grid lands it outside the PET's field of view, which is exactly how a volume
comes out uniformly −1000 HU (the `sitk.Resample` fill value for air). This is the cause,
and it should resolve once CT/PT/SEG come from one study.

Also mismatched: 3 of the 5 "smaller than the crop" patients
(`PETCT_32aa845af1`, `PETCT_ae96f738c0`, `PETCT_ca16242e89`) — same mechanism in a milder
form. The other 2 (`PETCT_2ad62beb52`, `PETCT_792d7b6ad0`) are correctly paired and are
just genuinely short-coverage scans.

### Why 59 and not 49 — resolved with TCIA metadata

Script: `scripts/tcia_study_inventory.py` (new, committed). One collection-wide `getSeries`
call (3,042 series), plus per-patient calls for the 81 multi-study patients. No image data.

The collection matches the brief exactly: **900 patients, 1,014 studies, 81 multi-study**.
Replaying the current `download_tcia.py` rule (largest ImageCount per modality) against that
metadata reproduces **49** mixed patients — your number, confirmed.

**Check 1 — are all 59 multi-study?** Yes: **59/59**. Zero single-study patients are
mismatched, so there is no second bug. (Stop condition not triggered.)

**Check 2 — tie evidence.** For **81/81** multi-study patients, *every* SEG series has
`ImageCount = 1` (a DICOM SEG is one multi-frame object). So `max(candidates, key=ImageCount)`
is a pure tie for the mask on every multi-study patient, and the winner is just whichever
series the API happened to list first. Worked examples:

| patient | CT (images per study) | PT (images per study) | SEG | on disk |
|---|---|---|---|---|
| `PETCT_32aa845af1` | 2007-01-26: **749** / 2007-06-15: 381 | 2007-01-26: **624** / 2007-06-15: 318 | both = 1 → **tie** | CT 2007-01-26, PT **2007-06-15**, SEG 2007-01-26 |
| `PETCT_73597f33fe` | 2007-05-17: **634** / 2007-02-10: 632 | 2007-05-17: **528** / 2007-02-10: 527 | both = 1 → **tie** | CT 2007-05-17, PT **2007-02-10**, SEG 2007-05-17 |
| `PETCT_d626611daf` | 2002-11-29: **793**, 4 others at ~441 | 2002-11-29: **661**, 4 others at ~368 | all 5 = 1 → **tie** | CT 2002-11-29, PT **2003-10-02**, SEG 2002-11-29 |

**But the tie story is not the whole explanation.** Those three rows also show a PT that is
*not* the largest series (318 images chosen over 624), which the max-ImageCount rule cannot
produce no matter how ties break. Replaying all three plausible rules against today's
metadata reproduces the on-disk study dates for only part of the collection:

| rule | patients whose 3 study dates match disk exactly | mismatches predicted |
|---|---|---|
| first series per modality in API order | 867 / 894 | 57 |
| largest ImageCount (current code) | 864 / 894 | 49 |
| largest FileSize | 829 / 894 | 21 |

No rule reproduces the disk exactly, on collection-wide *or* per-patient ordering. So the
archive on disk was **not produced by the current `download_tcia.py`** — most likely by the
older `download_all_patients.py` (still referenced in `.gitignore`), and/or TCIA's response
order is not stable between calls, which makes any order-dependent selection
non-reproducible after the fact.

Practical consequence: none. Every affected patient is in the re-download set, and the fixed
logic groups by StudyInstanceUID, so it is deterministic and ordering-independent.

### 897 audited vs 898 downloaded

`D:\data\dataset` holds **898 entries: 897 patient folders + 1 file**, `download_log.txt`.
The audit walks directories only, hence 897.

Separately, **3 of TCIA's 900 patients have no folder at all** — `PETCT_e9e1a391b5`,
`PETCT_ef9d41b836`, `PETCT_f01e83894d`. These are the three whose series-list fetch timed out
during the original download. 897 + 3 = 900.

---

## Step 2 — downloader / converter fix

### Download estimate (awaiting approval)

Connectivity: `download_tcia.py --check` succeeds **directly, no proxy needed**.

| set | patients | complete studies | size |
|---|---|---|---|
| all multi-study patients (covers all 59 mismatched) | 81 | 195 | 76.2 GB |
| the 3 incomplete patients | 3 | 3 | 1.49 GB |
| **total** | **84** | **198** | **77.7 GB** |
| _optional:_ the 3 never-downloaded patients | 3 | 3 | 1.46 GB |

Every study of every multi-study patient is complete (has CT+PT+SEG), so none are skipped.

**Time:** the existing 150 GB download took ~20 h → **7.5 GB/h**, so 77.7 GB ≈ **10–11 h**.

**Disk:** +78 GB of zips (`D:` has ~195 GB free). Converted NIfTI for the 114 net-new scans
adds roughly 8 GB.

**Yield:** those 81 patients contribute 81 scans today (one each); after repair they
contribute 195 → **+114 net new training scans**.

### Approved and running

87 patients: 81 multi-study + 3 incomplete + 3 never-downloaded (approved separately, for a
clean 900-patient basis). 79.2 GB.

| | |
|---|---|
| backup | 84 patients' flat-layout `CT/PT/SEG.zip` + `.done` → `D:\data\backup_mixed_studies\` (13.8 GB). Nothing deleted. |
| process | detached, survives this session — `Start-Process`, PID 44640 |
| log | `D:\data\reports\redownload.log` (errors: `redownload.err`) |
| patient list | `D:\data\reports\redownload_patients.txt` |
| sleep | disabled on AC (`powercfg /change standby-timeout-ac 0`); monitor still sleeps |
| resume | per-study `.done` markers — rerun the same command to continue |

### Deviation from the brief: study folder naming

The brief specifies `<StudyDate>_<first 8 chars of study UID>`. The first 8 characters of a
DICOM UID are the shared organisational root — `1.3.6.1.` for **every** study in this
collection — so every study of a patient would land in one directory and overwrite the
previous. The **last** 8 characters are used instead: `20030411_84294251`.

### Verified before launch

On `PETCT_c2ffda4725` (2 studies): both downloaded to separate directories, each with
CT+PT+SEG+`.done`; re-running skipped both (`studies already=2`); the audit reports 2 scans,
0 mismatches, and each scan's three modalities share one StudyInstanceUID.

### Code changes

**`download_tcia.py`** (`849b83a`) — series grouped by StudyInstanceUID, paired only within a
study; every complete study downloads as its own scan; per-study `.done` resume markers;
`.part` + zip integrity check, `--proxy` and `--check` unchanged; new `--patients` (list or
file). A single-study patient already present in the flat layout is left alone, since with one
study there is nothing to mix.

**`dicom_to_nifti.py`** (`0278b5e`) — every series now carries StudyInstanceUID/StudyDate. A
scan whose CT/PT/SEG do not share one study is **reported and skipped, never written**. A
multi-study patient converts to one directory per study. Verified both ways:

```
[1/2] PETCT_c2ffda4725/20030411_84294251
  ok PETCT_c2ffda4725/20030411_84294251: (400, 400, 242)  lesion_voxels=0
[2/2] PETCT_c2ffda4725/20030919_69927018
  ok PETCT_c2ffda4725/20030919_69927018: (400, 400, 242)  lesion_voxels=0

[1/1] PETCT_234f8427c0                          <- the old mixed archive, from backup
  [MISMATCH] PETCT_234f8427c0: CT/PT/SEG come from different studies
             (CT 20051203, PT 20041209, SEG 20041209) -- refusing
Done. converted=0 ... mismatched=1
```

No output file was written for the refused case.

**`check_study_pairing.py`** (`849b83a`) — audits both layouts, one CSV row per scan.

**`check_nifti_readable.py`** + **`check_pairing_visual.py`** (`6ddf0eb`) — see post-download
checks below.

### Post-download checklist (agreed additions)

1. **Old-NIfTI trap.** For every re-converted patient, move the old top-level
   `PET/CT_resample/tumorSeg.nii.gz` to `backup_mixed_studies\<pid>\nifti_old\`, then assert no
   patient holds both a top-level scan and study folders — `build_subject_index` rglobs
   `PET.nii.gz`, so such a patient would contribute the old mismatched scan as an extra one.
   Checker written and already passing on the current data (886 patients / 886 scans, 0
   offenders).
2. **SimpleITK sweep.** Read every NIfTI with SimpleITK, isolated per subprocess, since ITK
   can segfault rather than raise. Any file that fails gets re-converted.
3. **Visual check.** Coronal PET MIP + mask outline, CT MIP and an axial CT slice for the
   re-converted patients including `PETCT_234f8427c0`.

**Pre-repair baseline image** (`reports\pairing_check_before\PETCT_234f8427c0.png`) shows the
bug clearly: the PET is a normal whole-body scan with an obvious hot spot, the CT panel is
entirely black (100% of voxels ≤ −900 HU), and the mask is empty despite that hot spot —
because the SEG came from yet another visit. A good patient for contrast renders an uptake
ratio of 9.1× with the mask sitting on the hot spots.


## Step 2 — results

Download finished in **6 h 30 m**: `patients complete=87, partial=0, errors=0 | studies ok=199,
already=2, failed=0`. **195 study folders on disk = 195 expected**, no patient short.

**Correction to my estimate:** I quoted 77.7 GB; the actual download was **28.6 GB**. TCIA's
`FileSize` is the *uncompressed* DICOM size while `getImage` returns a ZIP (~2.7x, matching the
original 419 GB metadata → 150 GB on disk). The estimate was built on metadata bytes.

Conversion: **203 converted, 811 already present, 0 skipped, 0 mismatched, 0 errors.**

### Acceptance checks

| check | result |
|---|---|
| pairing audit on the new layout | **900 patients / 1014 scans, 0 mismatched, 0 incomplete** |
| no patient mixes flat + per-study layout | **pass** (asserted; see below) |
| SimpleITK readable, one file per subprocess | **3042/3042 readable, 0 failures** |
| 3 blank-CT patients now have real CT | **yes** — all now −1024..3071 HU (were uniformly −1000) |
| 5 short-coverage patients restored | yes (3 via re-conversion, 2 were already regenerated) |

**The layout assertion earned its keep.** Restoring the two short-coverage patients from
`excluded_nifti` with `mv` nested them inside their own freshly-converted folders
(`autopet_nifti/<pid>/<pid>/`), because the destination already existed. The assertion caught
it immediately; the stale copies went to backup. Those two are single-study, so conversion had
already regenerated them and they never needed restoring.

**Visual check** (`reports\pairing_check\`) — before/after on `PETCT_234f8427c0`, same
2004-12-09 visit: CT went from **100% air, entirely black** to a full skeleton and soft-tissue
anatomy (93% air, normal for a body in a whole-body field of view).

One scan was first flagged suspicious at 1.3x by the mask-mean/body-mean ratio.
It is correct: 11 separate lesions, each peaking **2x to 17x** the body median, contours sitting
exactly on the PET hot spots. The heuristic was averaging over mask edges and dividing by a mean
inflated by a bright bladder, so it now scores each lesion by its peak against the body median.

### Final dataset counts

| | |
|---|---|
| patients | **900** |
| scans | **1014** |
| patients with >1 scan | **81** (195 scans; 2 to 5 each) |
| tumour-positive scans | **501 (49.4%)** |
| tumour-free scans | 513 (50.6%) |
| tumour-positive patients | 489 (54.3%) |

900 patients / 1014 studies / **501 tumour-positive** reproduces the published AutoPET figures
exactly — an independent confirmation that the pairing repair is right.

## Step 3 — frozen split

`splits/autopet_v1.json` (committed, 84 KB), seed **2026**, unit = patient.

| set | patients | scans | tumour-positive patients | tumour-positive scans |
|---|---|---|---|---|
| train | 630 | 738 | 341 (54.1%) | 353 |
| val | 70 | 76 | 38 (54.3%) | 38 |
| test | **200** | **200** | 110 (55.0%) | 110 |
| _overall_ | 900 | 1014 | 489 (54.3%) | 501 |

Test is drawn only from single-study patients, so every test patient contributes exactly one
scan and all 81 multi-scan patients go to train/val. Stratification holds the positive fraction
within 0.7 points of the overall 54.3%. Verified: no patient in two sets, all 900 assigned,
1014 case IDs all unique. Multi-study case IDs look like
`PETCT_0410759456_20021107_05575415`.

## Step 4 — nnU-Net v2 baseline

### Environment (`D:\data\nnunet_venv`, separate from `petct`)

| package | version |
|---|---|
| **nnunetv2** | **2.8.1** |
| **torch** | **2.14.0+cu130** (CUDA available, RTX 5000 Ada, capability 8.9) |
| torchvision | 0.29.0+cu130 |
| numpy | 2.5.2 |
| SimpleITK | 2.5.6 |
| batchgeneratorsv2 | 0.3.5 |
| dynamic-network-architectures | 0.4.4 |
| blosc2 | 4.14.0 |

`pip install nnunetv2` replaced the CUDA torch with a CPU build (the same trap as in the
`petct` venv), so torch and torchvision were force-reinstalled from the cu130 index afterwards.
cu130 was chosen over cu128 because it carries torch 2.14, matching the torchvision nnU-Net
pulls in; the driver reports CUDA 13.2, so it is supported.

### Drive layout — Z: is a NETWORK drive

`Z:` is `\\10.228.48.113\Khurram`, a mapped network share, so nothing nnU-Net reads during
training goes there. Both C: and D: are local NVMe SSDs:

| variable | path | why |
|---|---|---|
| `nnUNet_raw` | `D:\data\nnunet\raw` | same volume as `autopet_nifti`, so the export hardlinks |
| `nnUNet_preprocessed` | `C:\nnunet_preprocessed` | local SSD, **386 GB free** — D: only has 150 GB |
| `nnUNet_results` | `D:\data\nnunet\results` | checkpoints only, small |

### Export (`scripts/export_nnunet.py`)

```
train: 738 cases -> imagesTr/labelsTr
val  :  76 cases -> imagesTr/labelsTr
test : 200 cases -> imagesTs/labelsTs
files: 3042 hardlinked, 0 copied        <- zero extra disk, originals untouched
numTraining = 814  (train 738 + val 76)
```

`dataset.json`: `channel_names {"0": "CT", "1": "PET"}` (CT gets nnU-Net's CT normalisation,
PET the default z-score), `labels {"background": 0, "tumour": 1}`, `file_ending ".nii.gz"`.
`splits_final.json` written with one fold, 738 train / 76 val, to be copied into
`nnUNet_preprocessed` after planning.

### Backup

`git bundle create D:\data\reports\petct_data-fix-nnunet.bundle main..data-fix-nnunet
main..workstation-local` — verified, 41 KB, contains both branch refs
(`abac067`, `a9739f3`).

### Fingerprint + plan (default `ExperimentPlanner`, `3d_fullres` only)

`nnUNetv2_extract_fingerprint -d 501 --verify_dataset_integrity` — **814/814 cases,
integrity verified, no errors** (5 min 23 s).

`nnUNetPlans.json`, 3d_fullres configuration:

| | |
|---|---|
| target spacing (z, y, x) | **3.0 × 2.0364 × 2.0364 mm** |
| median image size | 326 × 400 × 400 voxels |
| **patch size** | **128 × 128 × 128** |
| **batch size** | **2** |
| **stages** | **6** — features 32, 64, 128, 256, 320, 320; stride 1 then 2× five times |
| network | `PlainConvUNet`, InstanceNorm, LeakyReLU |
| normalisation | `CTNormalization` (ch0 = CT), `ZScoreNormalization` (ch1 = PET) |
| batch dice | True |

The target spacing comes out **identical to our conversion spacing**, because every scan is
already on its PET grid, so nnU-Net's resampling is effectively a no-op here. Our own pipeline
uses 3 × 2 × 2 mm, within 2% of this — so the nnU-Net comparison is not confounded by
resampling differences, only by the pipeline itself.

Contrast with our hand-configured CNN (`DESIGN_NOTES.md` §6): patch 96×128×128, batch 4, 6
stages down to a 3×4×4 bottleneck. nnU-Net independently chose an isotropic 128³ patch at
batch 2 — same network family, different geometry, which is exactly the pipeline difference
this baseline is meant to isolate.

### Fold 0 uses our split — confirmed by nnU-Net itself

```
Using splits from existing split file: C:\nnunet_preprocessed\Dataset501_AutoPET\splits_final.json
The split file contains 1 splits.
Desired fold for training: 0
This split has 738 training and 76 validation cases.
```

Independently verified before training: 0 overlap between fold-0 train and val, all 814 cases
present in the preprocessed directory, and **0 test patients leaked** into fold 0.

### Pilot (2 epochs, then stopped for approval)

| | |
|---|---|
| epoch time | 59.5 s, 51.9 s → **~56 s/epoch** |
| **projected 1000 epochs** | **~15.5 hours** |
| GPU utilisation | **97–98%** sustained |
| GPU memory | **7.8 GB of 32.8 GB** (24%) |
| power / temp | 223 W of 250 W, 78 °C |
| pseudo Dice | 0.0007 (epoch 0) → **0.0922** (epoch 1) |

Well under the ~4-day threshold, so no shorter trainer variant is needed — the default
`nnUNetTrainer` at 1000 epochs stands. `nnUNet_compile=f` was set to avoid torch.compile/triton
issues on Windows; utilisation is already at 98%, so nothing is left on the table.

Memory headroom is large (24% of the card), but batch size and patch size are the planner's
choices and changing them would stop this being the default-nnU-Net baseline.

### Full run — launched

| | |
|---|---|
| command | `nnUNetv2_train 501 3d_fullres 0` (fresh, no `--c`) |
| process | detached via `Start-Process`, **PID 52912**, survives this session |
| log | `D:\data\reports\nnunet_train_fold0.log`; nnU-Net's own log in `…/fold_0/training_log_*.txt` |
| started | 2026-09-28 23:31, ~15.5 h → expected finish ~15:00 on 2026-09-29 |
| pilot | kept at `…/fold_0_pilot/` (2-epoch log and checkpoint) |
| sleep | disabled on AC |

**Checkpoint policy, fixed before seeing any result:** test predictions will use
**`checkpoint_final.pth`** (nnU-Net's default, the weights after epoch 1000), *not*
`checkpoint_best.pth`. `checkpoint_best` is selected on this run's own validation set, so
reporting it as the test number would leak validation-based model selection into the headline
figure. `nnUNetv2_predict` uses `checkpoint_final` unless told otherwise, so this is also the
out-of-the-box behaviour.

### Preprocessed size — measured, not guessed

Storage is blosc2 (`.b2nd`). Measured on the first 16 completed cases: **0.068 GB per case →
~55 GB for 814**. Uncompressed those arrays would be ~470 MB/case (382 GB), so blosc2 is doing
about 8×. C: goes from 383 GB free to roughly **328 GB free** — far above the 40 GB floor, so
preprocessing proceeded without needing Z: at all.

## Step 4 result — Dataset501: "default nnU-Net, raw PET" baseline

Kept as-is and reportable in its own right: this is what the default nnU-Net pipeline does when
the PET channel is raw Bq/mL under nnU-Net's own per-image z-score. Dataset502 (SUV + global
normalisation) is built separately; 501 is not modified.


1000/1000 epochs in ~14.5 h, no errors, `checkpoint_final.pth` written. Final EMA pseudo-Dice
0.3633.

nnU-Net's own validation on the 76 val cases, reproduced independently by
`scripts/evaluate_predictions.py` (identical Dice, a good check on the evaluator):

| | val (76 cases) |
|---|---|
| Dice, tumour-positive cases | **0.1396** (median 0.0699) |
| complete misses | 0 |
| HD95 | 211 mm mean, 144 mm median |
| **tumour-free cases with any false positive** | **38 of 38 (100%)** |
| mean FP volume | **150 mL** per case (127 mL on tumour-free cases) |
| mean FN volume | 26 mL |

Published nnU-Net baselines on AutoPET reach roughly Dice 0.65 on tumour-positive cases with
false-positive volumes in the tens of mL. This is far below that, so it is reported as a
finding, not as "the nnU-Net number".

### What is NOT the cause

* **Data pairing** — the audit is clean, 0 mismatches over 1014 scans.
* **Label alignment** — checked in the *preprocessed* arrays nnU-Net actually trained on:
  lesions sit on strongly elevated PET (normalised PET inside lesion vs outside: thousands of
  times higher). The whole crop/resample/normalise chain is sound.
* **Channel order** — the fingerprint confirms channel 0 spans −1024..3071 HU (CT) and channel
  1 spans 0..289014 (PET), and the plan applies `CTNormalization` to 0, z-score to 1.
* **Split or leakage** — fold 0 verified as 738/76 with no test patient present.
* **Training stability** — loss fell smoothly, val tracked train with no overfitting, LR
  annealed normally, every epoch took 52 s.

### What is the cause: per-image z-score destroys PET's calibration

nnU-Net normalises any non-CT channel by **per-image z-score**. Measured across 60
tumour-positive cases, the median *normalised* PET intensity **inside the lesion**:

| percentile | normalised PET in lesion |
|---|---|
| p5 | **−0.12** (lesion at or below the image mean) |
| p25 | 1.36 |
| p50 | 4.83 |
| p75 | 9.22 |
| p95 | **18.86** |
| range | −0.22 … 25.69 |

The same tissue class lands anywhere from *below the image mean* to *26 standard deviations
above it*, depending on the patient. The network has no stable appearance to learn, so it
falls back on "whatever is relatively bright in this image" — which is why it fires on heart,
bladder and brown fat, and why every tumour-free case picks up false positives.

The CT channel, normalised with **dataset-wide** foreground statistics, is stable by contrast:
lesion-median p5 −1.92, p50 0.55, p95 0.68.

**Note on SUV.** Our PET is raw Bq/mL rather than SUV, but converting it would not by itself
fix this: per-image z-score is invariant to an affine rescale, since
`(a·x − a·μ)/(a·σ) = (x − μ)/σ` exactly, and SUV is Bq/mL times a per-patient constant. The fix
has to change the *normalisation scheme*, not just the units — e.g. convert to SUV and then
give the PET channel a fixed divisor (`noNorm`-style), which is what AutoPET nnU-Net pipelines
do. That needs the DICOM injected dose, patient weight and injection time, which the raw
archives have.

## Step 5 — evaluation script + our pipeline on the frozen split

## Step 6 — test-set prediction and evaluation

```
nnUNetv2_predict -i <raw>/Dataset501_AutoPET/imagesTs -o <predictions>/test_fold0 \
    -d 501 -c 3d_fullres -f 0 -chk checkpoint_final.pth
```

200/200 cases predicted, 42.7 s/case (~2 h 22 m), GPU 99% throughout, no errors.
`checkpoint_final.pth` as fixed in advance.

Scored with `scripts/evaluate_predictions.py` against `labelsTs`; per-case CSV at
`reports/eval_nnunet_test.csv`.

### Held-out test set — 200 patients, 110 tumour-positive, 90 tumour-free

| metric | value |
|---|---|
| **Dice, tumour-positive cases** | **0.1755** (median 0.1239), all 110 scored |
| complete misses (empty prediction) | **0** |
| **HD95** | **234.5 mm** mean, 190.3 mm median — covers 110 of 110 positives (100%) |
| **tumour-free cases with any false positive** | **90 of 90 (100%)** |
| mean FP volume, all cases | **151.3 mL** |
| mean FP volume, tumour-free cases | **127.2 mL** |
| mean FN volume, all cases | 15.7 mL |

Val and test agree closely (Dice 0.1396 vs 0.1755, FP 150 vs 151 mL), so this is a stable
property of the trained model, not a quirk of one split.

### The error signature: high recall, very low precision

The failure is entirely one-sided, and that is what points at normalisation rather than at a
model that simply cannot see lesions:

| | |
|---|---|
| complete misses | **0 of 110** — every tumour-positive case gets *something* predicted |
| mean FN volume | **15.7 mL** — relatively little true disease is missed outright |
| mean FP volume | **151.3 mL**, ~10x the FN volume |
| tumour-free cases with FP | **90 of 90** — not one clean negative |

So the model finds the lesions and then keeps going: it marks roughly ten times more spurious
tissue than it misses real tissue. A model that had failed to learn lesion appearance would
miss cases and under-segment; this one over-segments everywhere, including on 90 patients with
no disease at all. That is the signature of a decision rule along the lines of "bright relative
to this image" rather than "bright in absolute, calibrated terms" — exactly what a per-image
z-score leaves the network to work with. Val and test agree closely (Dice 0.1396 vs 0.1755,
FP 150 vs 151 mL), so it is a stable property of the model, not a split quirk.

**This is not a usable baseline number.** For scale, a whole-body FDG PET/CT study contains
roughly 1–2 L of tissue that is bright on PET for entirely physiological reasons (brain,
heart, bladder, kidneys, brown fat), and the model is assigning ~127 mL of it to "tumour" on
every single tumour-free patient. Published nnU-Net on AutoPET reaches roughly Dice 0.65 with
false-positive volumes in the tens of mL.

The cause is the PET normalisation analysed above, not the data repair, the split, the export
or the training run — all of which check out. The next experiment is the one that tests it:
convert PET to SUV and give that channel a fixed divisor instead of a per-image z-score.

## Step 7 — isolating the cause on CPU, before spending another 15 h

100 training patients, voxels pooled and restricted to body (CT > −500 HU; air is trivially
separable and would flatter every scheme). 2.8 M sampled voxels, 447 M after weighting.
A single global threshold is a far weaker model than a U-Net, so the absolute numbers are low
in every arm — the *differences* are the result.

| scheme | pooled ROC AUC | best-threshold Dice | threshold | recall | FPR |
|---|---|---|---|---|---|
| per-image z-score (what nnU-Net applied) | **0.8249** | **0.1442** | z = 10.75 | 0.323 | 0.0074 |
| raw Bq/mL, global scaling | **0.8791** | **0.1755** | 12018 Bq/mL | 0.382 | 0.0070 |
| SUVbw, global scaling | **0.8794** | **0.1677** | SUV = 4.13 | 0.383 | 0.0075 |

Two results, and they point in different directions:

* **Global scaling clearly beats per-image z-scoring**: AUC 0.879 vs 0.825, and at essentially
  the same false-positive rate it recovers 38% of lesion voxels instead of 32%.
* **SUV vs raw Bq/mL is a tie**: ΔAUC 0.0003. Raw even edges SUV on best-threshold Dice
  (0.1755 vs 0.1677) at a slightly lower FPR.

So the per-image z-score is the thing that was hurting, and on *this* cohort the per-patient SUV
calibration adds nothing measurable on top of simply using a fixed scale.

Per-patient reference values:

| quantity | p5 | p50 | p95 | p95/p5 |
|---|---|---|---|---|
| median body SUV | 0.471 | 0.590 | 0.716 | **1.5x** |
| median body z | 0.721 | 1.238 | 1.636 | **2.3x** |
| median lesion SUV | 0.569 | 2.267 | 7.430 | 13.1x |
| median lesion z | 1.077 | 6.182 | 19.279 | 17.9x |

Normal body tissue sits at SUV ≈ 0.59 within a 1.5x band across patients — a stable anchor, so
a fixed decision boundary transfers. The same tissue under per-image z-scoring drifts over
2.3x, so no fixed boundary holds. Lesion spread is wide in both (genuine biological variation)
but consistently tighter in SUV.

### Why SUV adds nothing here (and so is not being built)

The SUV factor is nearly constant across this cohort, because the acquisition protocol is
tightly controlled:

| quantity | p5 | p50 | p95 | p95/p5 |
|---|---|---|---|---|
| injected dose | 292 MBq | 312 | 328 | **1.12x** |
| uptake delay | 55.5 min | 60.1 | 65.4 | **1.18x** |
| patient weight | 49 kg | 79 | 112 | 2.29x |
| **SUV factor** | 2.62e-4 | 3.68e-4 | 5.13e-4 | **1.96x** (CV 22%) |

Dose and uptake time are almost fixed, so SUV is very nearly a single global rescale of Bq/mL
and cannot reorder voxels pooled across patients. The per-image z-score, by contrast, divides
each scan by its own standard deviation — driven by bladder filling, body size and field of
view — which is what reorders them.

All 100 patients had complete, valid SUV tags (`Units = BQML`, `DecayCorrection = START`, dose,
half-life 6586.2 s, weight, injection time).

**SUV is still what Dataset502 uses**, despite the tie, for reasons the AUC does not capture:
raw Bq/mL only behaves like SUV where dosing is weight-based, which is a property of this site
rather than of PET; the plausibility checks in the work plan are written in SUV; and SUV is the
field standard, so thresholds and figures are readable by anyone else. The tie is worth a
sentence in the write-up precisely because it says something about *this dataset's* acquisition
protocol, not about PET normalisation in general.

### Dataset502: same data, global normalisation

The fix needs no re-conversion and no new NIfTI — only a different `dataset.json`:

```json
"channel_names": {"0": "CT", "1": "CT"}
```

`"CT"` selects `CTNormalization`, which clips to the dataset-wide foreground 0.5/99.5
percentiles and then subtracts a **dataset-wide** mean and divides by a **dataset-wide** std
(`get_normalization_scheme` looks each channel up independently, so the repeated name is fine).
Every patient therefore gets the identical transform, which is the property that matters. The
name is a misnomer for a PET channel; the scheme is what is being selected.

Chosen over `noNorm` on clipped values because the clip bounds and scale come from the data's
own statistics rather than hand-picked constants, and because it keeps everything inside
nnU-Net's standard machinery — Dataset502 then differs from Dataset501 in exactly one variable.

Known trade-off: the clip at the foreground p99.5 (~75,500 Bq/mL ≈ SUV 23) puts a very hot
bladder and a very hot lesion at the same ceiling. The CT channel still separates them
anatomically, and clipping PET is standard practice.

### Cost of the run, for the record

| stage | time |
|---|---|
| fingerprint + integrity | 5 min |
| planning | 2 min |
| preprocessing (814 cases, 62.5 GB) | 20 min |
| training, 1000 epochs | 14 h 30 m |
| built-in validation (76 cases) | 12 min |
| test prediction (200 cases) | 2 h 22 m |
| evaluation (200 cases) | 18 min |

---

## Commits

| Hash | Branch | Message |
|---|---|---|
| a9739f3 | `workstation-local` | Pad undersized body-crop volumes before the random training crop |
