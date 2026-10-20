# Does the MAE pretraining help? Dataset505, pretrained against random init

**Question.** Dataset505 reached test Dice 0.6421 from MAE-pretrained weights. How
much of that is the pretraining, and how much is the architecture, the data and
nnU-Net's recipe? This is the control: the same run with random initialisation.

Identical in every respect except the initialisation — same preprocessed data,
same `nnUNetPlans_MAE`, same `nnUNetTrainer_500epochs` schedule, same fold 0 of the
same `splits_final.json`, 738 train / 76 val. The only change is omitting
`-pretrained_weights`.

To stop the control overwriting the pretrained run — nnU-Net derives its output
folder from the trainer name — the control runs under
`nnUNetTrainer_500epochs_scratch`, a subclass of `nnUNetTrainer_500epochs` that adds
nothing but a name. It lives in `nnunet_ext/` and is found through nnU-Net's
`nnUNet_extTrainer` variable, so nothing was written into site-packages.

Verified before the run started: split 738/76, **no** pretrained-weights banner
anywhere in the log, and the first encoder conv differing from the MAE checkpoint
(max |difference| 0.99; σ 0.196 against 0.207). Verified after: the pretrained run's
two checkpoints and all 326 of its mapped predictions hash **unchanged**.

## Training

| | MAE pretrained | from scratch |
|---|---|---|
| Best EMA pseudo-Dice | **0.7857** (epoch 490) | 0.7620 (epoch 474) |
| Mean EMA, last 50 epochs | **0.7717** | 0.7400 |
| Wall clock | 8 h 16 m | 8 h 17 m |
| Epochs where it led | **494 / 500** | 6 |

Epoch at which each run first reached a given EMA pseudo-Dice:

| EMA | pretrained | scratch | epochs saved |
|---|---|---|---|
| 0.40 | 17 | 24 | +7 |
| 0.50 | 25 | 43 | +18 |
| 0.60 | 43 | 118 | **+75** |
| 0.65 | 98 | 196 | **+98** |
| 0.70 | 290 | 359 | +69 |
| 0.74 | 345 | 444 | +99 |
| 0.7620 | 447 | **never** | — |

The gap narrows but does not close: +0.078 at epoch 50, +0.044 at 250, +0.028 at 499.

Plot: `curves_505_pretrained_vs_scratch.png`. The EMA is recomputed from the
per-epoch pseudo-Dice with nnU-Net's own rule (`0.9·prev + 0.1·current`) because the
logs print the EMA only when it improves; the recomputation reproduces each run's
logged best exactly (0.7857 and 0.7620).

## Held-out results

### Test — 200 patients, 110 tumour-positive

| metric | pretrained | scratch | change |
|---|---|---|---|
| Dice | **0.6421** | 0.6179 | ×0.96 |
| median Dice | 0.7580 | 0.7420 | ×0.98 |
| positives scoring Dice 0 | 4 | 5 | +1 |
| HD95 mm (mean) | **128.0** | 146.7 | +15% |
| median HD95 mm | **39.0** | 60.2 | +54% |
| FP volume mL | 15.94 | **15.40** | −3% |
| FN volume mL | **5.45** | 7.20 | +32% |

### Validation — 76 patients, 38 tumour-positive

| metric | pretrained | scratch | change |
|---|---|---|---|
| Dice | **0.6963** | 0.6809 | ×0.98 |
| median Dice | 0.7972 | 0.7765 | ×0.97 |
| HD95 mm (mean) | 79.7 | **73.9** | −7% |
| FP volume mL | 10.21 | **9.38** | −8% |
| FN volume mL | 4.28 | **4.13** | −4% |

### Fit check — 50 training cases, 18 tumour-positive

| | pretrained | scratch |
|---|---|---|
| train Dice | **0.7461** | 0.7141 |
| train − test | +0.1040 | +0.0962 |

Both now fit their training data better than held-out data, by a similar margin.
Whatever the pretraining contributes, it is not a change in overfitting behaviour.

## Paired bootstrap

Both models are scored on the same patients, so the per-patient *differences* are
resampled — 10,000 draws, seed 2026, percentile CI.

### Test Dice, 110 tumour-positive patients

| | |
|---|---|
| paired difference (scratch − pretrained) | **−0.0243** |
| 95% CI | **[−0.0510, +0.0021]** — straddles 0, barely |
| median difference | −0.0078 |
| per-patient | pretrained better on **66**, scratch on 40, tied 4 |
| sign test | **p = 0.016** |

### Test false-positive volume, all 200 patients

| | |
|---|---|
| paired difference | −0.54 mL (scratch lower) |
| 95% CI | [−4.09, +3.21] — straddles 0 |
| per-patient | scratch better on 86, worse on 90, tied 24 |
| sign test | p = 0.82 |

### Validation Dice, 38 tumour-positive patients

| | |
|---|---|
| paired difference | −0.0154 |
| 95% CI | [−0.0423, +0.0166] — straddles 0 |
| per-patient | pretrained better on 24, scratch on 12, tied 2 |
| sign test | p = 0.064 |

## Reading this honestly

**The two tests on test Dice disagree, and the disagreement is the finding.** The
bootstrap CI on the *mean* difference includes zero (upper bound +0.0021, so only
just). The sign test on *which model wins per patient* does not (p = 0.016, 66
against 40). Both are correct: per-patient Dice differences are enormous next to
the mean effect — individual cases swing by +0.68 and −0.64 — so 110 patients
cannot pin a 0.024 mean down, while the direction of the difference is consistent
enough to show up.

So: **the pretraining helps, modestly and consistently, and this test set is too
small to put a confident number on the size.** Claiming "pretraining gives +0.024
Dice" would overstate what 110 patients support; claiming "no difference" would
ignore that it wins on 60% of them, led on 494 of 500 epochs, and ended higher on
every aggregate except false-positive volume.

**Where the benefit actually sits.** Mostly in convergence speed: the pretrained run
reached EMA 0.65 at epoch 98 against 196, half the training. It also ends higher —
+0.032 EMA averaged over the last 50 epochs, and the scratch run never reaches the
pretrained best at any point in 500 epochs — but that end-state advantage is the
smaller of the two effects and the one the test set cannot resolve.

**False positives are not improved by pretraining.** 15.4 against 15.9 mL, p = 0.82,
and the scratch model is better on 86 patients against 90. The residual
false-positive load is a property of the task and the recipe, not the
initialisation, and it is the obvious target for the post-processing work.

**The comparison is not clean, and it favours the pretrained model.** The MAE
pretraining corpus (Liu et al. 2026) includes AutoPET, so the pretrained encoder has
almost certainly already seen the images in our test and validation sets — without
their labels, but seen them. That is unlabelled transductive exposure the scratch
model does not get, and it plausibly accounts for some unknown part of the
difference measured here. The honest reading is that +0.024 is an *upper* bound on
what the pretraining would buy on genuinely unseen data, which makes the already
weak effect weaker still. Nothing in this setup can separate the two, because the
pretraining corpus cannot be un-seen; only pretraining on a corpus with our test
patients excluded would settle it.

**Scope.** One fold, one seed, one architecture. A second seed would show how much
of the 0.024 is run-to-run noise; that is the cheapest next thing worth doing if
this number has to be defended.

## Artifacts

| | |
|---|---|
| Scratch trainer | `nnunet_ext/nnUNetTrainer_500epochs_scratch.py` |
| Training log | `D:\data\reports\nnunet_505_scratch_fold0.log` |
| Checkpoints | `...\Dataset505_AutoPET_MAEprep_v2\nnUNetTrainer_500epochs_scratch__nnUNetPlans_MAE__3d_fullres\fold_0` |
| Per-case scores | `reports\eval_505s_{test,val,train}.csv` |
| Bootstrap output | `reports\bootstrap_{test_dice,test_fp,val_dice}.txt` |
| Curve plot | `reports\curves_505_pretrained_vs_scratch.png` |
| Pre-run hashes | `reports\hashes_505_before_scratch.csv` |
