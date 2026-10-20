# Label efficiency: 10%, 30% and 100% of the training patients, with and without pretraining

**Questions.** How far does performance fall when only a fraction of the labels
exist, and does MAE pretraining pay more when labels are scarce?

Six runs on Dataset505's corrected labels, all 500 epochs, same schedule, same
plans, same validation set, same test set. Only the training subset and the
initialisation differ.

## The subsets

Fractions of the 630 **training patients**, not of the scans: repeat visits are
highly correlated, so a subset takes every scan of a chosen patient or none.
Stratified on tumour-positive patient status, seed 20261005. Subsets are prefixes
of one seeded ordering per stratum, so 10% ⊂ 30% ⊂ 100% — verified, not assumed.

| | patients | scans | tumour-positive scans | positive-patient fraction |
|---|---|---|---|---|
| 10% | 63 | 71 | 34 | 0.5397 |
| 30% | 189 | 220 | 106 | 0.5397 |
| 100% | 630 | 738 | 353 | 0.5413 |

Added to `splits_final.json` as folds 1 and 2, with every pre-existing fold
verified byte-identical afterwards and all folds sharing fold 0's validation list.

## Test — 200 patients, 110 tumour-positive

| run | Dice | median | HD95 mm | FP mL | FN mL | tumour-free with FP | empty |
|---|---|---|---|---|---|---|---|
| 100% pretrained | **0.6421** | 0.7580 | 128.0 | 15.94 | 5.45 | 63/90 | 1 |
| 100% scratch | 0.6179 | 0.7420 | 146.7 | 15.40 | 7.20 | 69/90 | 2 |
| 30% pretrained | 0.6213 | 0.7228 | 128.9 | 8.89 | 16.72 | 59/90 | 3 |
| 30% scratch | 0.6101 | 0.7061 | 128.3 | 6.97 | 16.58 | 52/90 | 3 |
| 10% pretrained | 0.5667 | 0.6396 | 140.3 | 5.79 | 24.00 | 61/90 | 2 |
| 10% scratch | 0.5540 | 0.6058 | 157.7 | 9.80 | 19.49 | 69/90 | 2 |

## Validation — 76 patients, 38 tumour-positive

| run | Dice | median | HD95 mm | FP mL | FN mL | tumour-free with FP | empty |
|---|---|---|---|---|---|---|---|
| 100% pretrained | **0.6963** | 0.7972 | 79.7 | 10.21 | 4.28 | 26/38 | 0 |
| 100% scratch | 0.6809 | 0.7765 | 73.9 | 9.38 | 4.13 | 27/38 | 1 |
| 30% pretrained | 0.6746 | 0.7642 | 79.1 | 8.70 | 5.63 | 22/38 | 0 |
| 30% scratch | 0.6531 | 0.7467 | 85.4 | 7.45 | 7.57 | 21/38 | 1 |
| 10% pretrained | 0.6257 | 0.7304 | 82.5 | 6.36 | 8.69 | 22/38 | 2 |
| 10% scratch | 0.5925 | 0.7059 | 114.4 | 9.29 | 9.18 | 26/38 | 1 |

## 1. Label efficiency — the robust result

| labelled scans | test Dice (pretrained) | % of full | test Dice (scratch) | % of full |
|---|---|---|---|---|
| 71 (10%) | 0.5667 | 88.3% | 0.5540 | 89.7% |
| 220 (30%) | 0.6213 | **96.8%** | 0.6101 | **98.7%** |
| 738 (100%) | 0.6421 | 100% | 0.6179 | 100% |

**The curve is very flat.** 30% of the labels recovers ~97% of full-data Dice; the
remaining 518 annotated scans buy 0.021 Dice. 10% still gets ~88%. The same shape
holds on validation (0.6257 → 0.6746 → 0.6963) and for both initialisations, so it
is not an artifact of one split or one init.

For an annotation-budget decision this is the number that matters, and it says the
marginal value of labels past a few hundred scans is small.

The *way* small-label models degrade is consistent: they under-segment rather than
hallucinate. Going 100% → 10% (pretrained), false-negative volume rises 5.45 → 24.00
mL while false-positive volume *falls* 15.94 → 5.79 mL, and pooled voxel recall
drops from a much higher base while precision stays high.

## 2. Does pretraining help more when labels are scarce? No resolvable effect.

Paired bootstrap, 10,000 resamples, seed 2026, over the same patients.

### Dice, test (110 tumour-positive)

| | 10% | 30% | 100% |
|---|---|---|---|
| pretrained − scratch | +0.0127 | +0.0112 | +0.0243 |
| 95% CI (scratch − pretrained) | [−0.0319, +0.0054] | [−0.0363, +0.0125] | [−0.0510, +0.0021] |
| per-patient (pre / scr / tie) | 62 / 42 / 6 | 57 / 48 / 5 | 66 / 40 / 4 |

### Dice, validation (38 tumour-positive)

| | 10% | 30% | 100% |
|---|---|---|---|
| pretrained − scratch | +0.0332 | +0.0215 | +0.0154 |

**Every Dice confidence interval straddles zero, at every label fraction.**
Validation shows a tidy monotone decrease in the gap as labels increase
(+0.0332 → +0.0215 → +0.0154) — exactly the textbook story — but test shows no
trend at all (+0.0127 → +0.0112 → +0.0243), and the two splits disagree about
direction. With all six CIs including zero, the validation "trend" is not evidence;
it is three noisy numbers that happen to fall in order.

The pretrained model does win more patients than it loses in all six comparisons,
which is a consistent signal of *something*. But the effect is small enough that
200 test patients cannot size it, and 500 epochs is long enough for a scratch
network to largely catch up.

## 3. The 10% false-positive result did not replicate — withdrawn

The 10% comparison produced the one statistically solid finding in this project:
pretraining cut false-positive volume 41% (9.80 → 5.79 mL), CI excluding zero,
sign test p = 0.0007. With 30% now run, that result stands alone and the direction
reverses either side of it:

| FP volume, test | difference (scratch − pretrained) | 95% CI | verdict |
|---|---|---|---|
| 10% | **+4.01 mL** (pretrained lower) | **[+1.86, +6.54]** | excludes 0, p = 0.0007 |
| 30% | −1.92 mL (**scratch** lower) | [−4.57, +0.01] | straddles 0 |
| 100% | −0.54 mL (**scratch** lower) | [−4.09, +3.21] | straddles 0, p = 0.82 |

At two of three label fractions the scratch model has the lower false-positive
volume. Only at 10% does pretraining win, and only there is anything significant.
Lesion detection at 30% is likewise flat (0.6626 against 0.6610, CI
[−0.0197, +0.0145]), where at 10% the scratch model detected significantly more.

**One significant result out of three comparisons, pointing the opposite way to the
other two, is what noise looks like.** The 10% false-positive finding should not be
quoted as evidence that pretraining suppresses false positives. It is withdrawn.

For the record, the 10% comparison was checked carefully before this: the pretrained
model was *not* simply predicting less (total predicted volume 70.24 against 71.78
mL, CI straddling zero), its precision was genuinely better, and it placed 56% less
volume on tumour-free patients. All of that was true, and all of it failed to
reappear at 30%.

## 4. Training curves

`curves_505_label_efficiency.png` — all six runs. EMA recomputed from per-epoch
pseudo-Dice with nnU-Net's rule and checked against each run's logged best (0.7857,
0.7620, 0.7576, 0.7373, 0.6678, 0.6571; all exact).

| best EMA pseudo-Dice | 10% | 30% | 100% |
|---|---|---|---|
| pretrained | 0.6678 | 0.7576 | 0.7857 |
| scratch | 0.6571 | 0.7373 | 0.7620 |

**Do not rank models by these.** EMA pseudo-Dice is per training patch inside the
epoch loop; the tables above are full-volume Dice after sliding-window inference
and map-back. The two disagree: EMA makes the 100% gap (+0.0237) larger than the
10% gap (+0.0107), while real validation Dice says the opposite. The curves are for
convergence shape only.

## Caveats

The MAE pretraining corpus (Liu et al. 2026) includes AutoPET, so the pretrained
encoder has already seen the test and validation images without their labels. That
favours the pretrained model in every comparison here, and more so at small label
fractions. Any pretraining effect measured here is an upper bound on what the same
initialisation would give on genuinely unseen patients.

One fold and one seed per cell, and one draw of patients per subset. With six cells
and differences of 0.01–0.02 Dice, run-to-run and draw-to-draw variation is
plausibly as large as everything being compared. The label-efficiency curve is
robust because its effects are 0.07–0.08 Dice; the pretraining comparison is not,
because its effects are a quarter of that.

## Artifacts

| | |
|---|---|
| Subsets | `splits/autopet_v1.json` → `splits.train.subsets.{10pct,30pct}` |
| Training logs | `reports\nnunet_505_{10,30}pct_{pretrained,scratch}_fold{1,2}.log` |
| Per-case scores | `reports\eval_505{,s}_{10,30}pct_{test,val}.csv` |
| Detection metrics | `reports\detect_505{,s}_{10,30}pct_test.csv` |
| Bootstraps | `reports\bootstrap_{10,30}pct_*.txt` |
| Curve plot | `reports\curves_505_label_efficiency.png` |
