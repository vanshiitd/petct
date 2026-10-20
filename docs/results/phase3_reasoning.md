# Phase III: does reasoning beat confidence, and does refinement help?

> **Correction, 2026-10-08.** This report's Experiment 4 numbers were measured
> against an M0 that was weaker than the published 505 baseline (Dice 0.6347 /
> FP 19.10 mL against 0.6421 / 15.94 mL). The cause was the rule used to fuse
> the 8 flips: averaging probabilities instead of logits predicts ~10% more
> volume. M0 has been regenerated with logit fusion and now reproduces the
> baseline (Dice 0.6425 / FP 15.96 mL). **Every Experiment 4 figure below has
> been recomputed against that corrected baseline**, and both the headline
> false-positive reduction and the H2 recovery result shrank substantially. The
> superseded figures are kept in the text, marked, because the difference
> between them is the point. See `phase3b_cv.md` for the full reconciliation.

Branch `phase3-reasoning` from `phase2-uncertainty`. No training. Every model,
threshold and rule in this report was fitted on validation with patient-grouped
cross-validation; test was scored once, at the end.

---

## Step 1: M0 and its uncertainty

M0 is the Dataset505 MAE-pretrained model alone — the 2-model ensemble was
worse on test in Phase II. Its existing predictions were confirmed to use
nnU-Net's 8-flip mirroring (`inference_allowed_mirroring_axes = (0,1,2)`, no
`--disable_tta`). Its 8 flips were re-run for val, test and the 50 fit-check
cases: 326 cases, ~11 s each, 3.7 GB.

**P_M0 > 0.5 does not exactly reproduce the existing 505 predictions, and the
difference is systematic.** Median Dice against them is 0.983 (val) / 0.989
(test), but total predicted volume is **+1.45% / +3.30%**, and a handful of
cases disagree badly, all on marginal sub-10 mL blobs. The cause is structural:
P_M0 is the mean of 8 softmax maps, while nnU-Net averages logits and softmaxes
once, so one strongly negative flip can veto there but only dilutes here.

M0 is therefore defined as `P_M0 > 0.5` and **evaluated afresh**, so the
candidate set, the probability map and the baseline are one object and every
Experiment 4 delta is attributable to the actions rather than to a mismatch:

| | Dice | FP mL | FN mL | tumour-free with FP |
|---|---|---|---|---|
| published 505 pretrained (Phase I/II) | 0.6421 | 15.94 | 5.45 | 63/90 |
| **M0 as used here** | **0.6347** | **19.10** | 4.93 | 63/90 |

### How much did the second model add?

Component AUROC for separating FP from TP, same components and labels, only
the score changing:

| | val | test |
|---|---|---|
| U_stab, 2 models | 0.8531 | 0.7941 |
| U_stab_single, M0 only | 0.8444 | 0.7724 |
| **gain from the second model** | **+0.009** | **+0.022** |

Spearman between the two scores is 0.84. **Half the inference buys ~97% of the
signal.** Everything below uses the single-model version.

---

## Step 2: candidates

6-connectivity throughout, matching the evaluator. Components under 5 voxels
are not candidates — at t_low = 0.1 a whole-body map fragments into tens of
thousands of single-voxel specks (33,405 dropped on val, 86,262 on test) which
no action could sensibly take and which would dominate every statistic. Small
predicted components are still *kept* in M0; they are simply never removed.

| | val | test |
|---|---|---|
| predicted components | 476 TP / 233 FP | 970 TP / 698 FP |
| recovery candidates | 29 recoverable / 283 spurious | 76 / 796 |

Recovery is a rare class: **9.3% positive on val, 29 positives in total.** That
is a small basis for fitting and is flagged throughout.

---

## Experiment 3: error detection

### Predicted components — is this blob a false positive?

| detector | val CV AUROC | val CV AP | test AUROC | test AP |
|---|---|---|---|---|
| (a) confidence (P, U_ent) | 0.8171 ± 0.061 | 0.652 ± 0.186 | 0.8009 | 0.6909 |
| (b) + U_stab_single | 0.8651 ± 0.050 | 0.756 ± 0.084 | 0.7886 | 0.7141 |
| (c) reasoning only | 0.7497 ± 0.075 | 0.582 ± 0.163 | 0.7420 | 0.6251 |
| **(d) everything** | **0.8713 ± 0.029** | 0.769 ± 0.061 | **0.8139** | **0.7446** |
| (e) hand-written rules | 0.7386 (val fit) | 0.523 | 0.6364 | 0.5072 |
| (d) as gradient boosting | 0.8800 ± 0.038 | 0.776 ± 0.072 | 0.8503 | 0.8210 |

Paired bootstrap on test, (d) − (a): **+0.0130, 95% CI [−0.0070, +0.0313] —
straddles zero.**

Note (b) beats (a) by 0.048 on val CV and *loses* to it by 0.012 on test. With
76 patients to fit on, a 0.05 val gain is not worth believing.

### Recovery candidates — is this sub-threshold blob a real missed lesion?

| detector | val CV AUROC | test AUROC | test AP |
|---|---|---|---|
| (a) confidence | 0.6507 ± 0.090 | 0.5900 | 0.1077 |
| (b) + U_stab_single | 0.6701 ± 0.088 | 0.6566 | 0.1227 |
| **(c) reasoning only** | 0.6537 ± 0.137 | **0.7163** | **0.1761** |
| **(d) everything** | **0.7473 ± 0.135** | **0.7374** | 0.1701 |
| (e) hand-written rules | 0.4697 | 0.4654 | 0.0833 |
| (d) as gradient boosting | 0.6537 ± 0.216 | 0.7782 | 0.2427 |

Paired bootstrap on test, (d) − (a): **+0.1474, 95% CI [+0.0722, +0.2214] —
excludes zero.**

### Coefficients, detector (d), standardised

Predicted components, positive = more likely FP. Stable terms only
(|coefficient| larger than its spread across CV folds):

| feature | coef | ± across folds |
|---|---|---|
| ustab_max | **+1.411** | 0.189 |
| suv_peak | **−1.111** | 0.280 |
| suv_max | +0.749 | 0.334 |
| uent_mean | +0.726 | 0.201 |
| frac_spleen | −0.650 | 0.166 |
| p_mean | +0.599 | 0.217 |
| log_volume | −0.422 | 0.129 |
| frac_fluid_hu | +0.355 | 0.081 |
| frac_fat_hu | +0.327 | 0.052 |

Eight of twenty-one coefficients are smaller than their fold-to-fold spread and
are not interpretable; `suv_peak` and `suv_max` take opposite signs, which is
collinearity between two measures of the same thing rather than two findings.
The signs that do hold up are sensible: unstable under flips, low peak SUV,
small, fluid- or fat-range HU ⇒ false positive.

Recovery candidates, positive = recoverable: `hu_mean` +0.848, `frac_physio`
−0.648, `uent_mean` +0.639, `frac_fat_hu` −0.623, `suv_max` −0.534 — i.e. soft
tissue, away from physiological uptake, not fat.

### Case-level quality score Q

Q = volume-weighted mean P(not FP) over M0's components (eq. 16).

| | Q (this work) | Phase II uncertainty-only |
|---|---|---|
| vs Dice | **ρ = +0.736** | ρ = 0.74 (vs 1 − Dice) |
| vs FP volume | ρ = −0.210 | ρ = 0.72 (summed U_stab) |

Q matches the uncertainty-only score for ranking case *quality* and is far
worse for false-positive *volume* — the same lesson as Phase II: a mean tracks
quality, a sum tracks quantity, and Q is a mean.

---

## Experiment 4: single-pass refinement

Thresholds tuned on val. Dice point τ_A = 0.85, τ_B = 0.15; conservative point
τ_A = 0.50, τ_B = 0.30 (losing 0.99% of val true-positive volume, inside the 1%
budget).

### Test, all evaluator metrics (corrected baseline)

Thresholds retuned on val against the corrected M0: Dice point tau_A = 0.85,
tau_B = 0.15; conservative point tau_A = 0.75, tau_B = 0.20 (keeping 99.97% of
val true-positive volume).

| variant | Dice | median | IoU | HD95 mm | ASSD mm | FP mL | FN mL | lesion sens | FP lesions | tumour-free with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 (= published 505) | 0.6425 | 0.7584 | 0.5305 | 128.3 | 43.8 | 15.96 | 5.37 | 0.552 | 4.28 | 62/90 |
| **A remove [cons]** | **0.6559** | **0.7672** | **0.5454** | **106.4** | **32.6** | **11.71** | 6.17 | 0.518 | **2.91** | **47/90** |
| A+B [cons] | 0.6544 | 0.7672 | 0.5443 | 106.5 | 33.9 | 11.73 | 6.15 | 0.522 | 3.12 | 48/90 |
| A+B [dice] | 0.6410 | 0.7543 | 0.5311 | 124.4 | 39.6 | 14.71 | 5.58 | 0.540 | 3.60 | 52/90 |

For the record, the same table against the old weaker M0 read: M0 0.6347 /
FP 19.10, A remove 0.6575 / FP 3.43 / 40 of 90 tumour-free cleared. **The 82%
false-positive reduction reported there was largely removing excess the fusion
rule had introduced, not excess the model produced.** Against the correct
baseline the reduction is 27%.

### Section-23 numbers, against the corrected M0

| variant | Δ Dice | 95% CI | improved | unchanged | degraded | Δ HD95 | tumour-free cleaned |
|---|---|---|---|---|---|---|---|
| A remove [cons] | +0.0134 | [−0.0067, +0.0356] | 50 | 33 | 27 | **−19.9 mm** | 15 |
| A+B [cons] | +0.0119 | [−0.0077, +0.0339] | 49 | 30 | 31 | −19.9 mm | 15 |
| A+B [dice] | −0.0015 | [−0.0150, +0.0090] | 41 | 37 | 32 | −1.9 mm | 11 |

Every Δ Dice interval still straddles zero.

### Lesions lost and recovered, by size (corrected baseline)

| variant | < 0.1 mL | 0.1–0.5 | 0.5–1 | ≥ 1 mL |
|---|---|---|---|---|
| A remove [cons] | −1 / +0 | −17 / +0 | −15 / +0 | **−27 / +0** |
| A+B [cons] | −1 / +0 | −17 / +3 | −15 / +2 | −27 / +2 |
| A+B [dice] | −0 / +0 | −8 / +3 | −9 / +2 | −15 / +5 |

The cost is less than half what the weak baseline suggested — 60 lesions lost
rather than 146, 27 of them ≥ 1 mL rather than 72 — and the character of the
refinement changes with it. At the gentler threshold the removal improves Dice,
HD95 *and* ASSD together (43.8 → 32.6 mm), where against the weak baseline ASSD
had worsened 46 → 70 mm. Lesion sensitivity still falls, 0.552 → 0.518.

### Case quality score Q, corrected

| | Q | Phase II uncertainty-only |
|---|---|---|
| vs Dice | ρ = +0.711 | ρ = 0.74 |
| vs FP volume | ρ = −0.144 | ρ = 0.72 |

### Examples

`phase3_examples.png`. Top: a tumour-free patient, M0 predicting 7 components
and 293.9 mL of false positive, reduced to 1 component against an empty ground
truth. Bottom: a patient with extensive disease where recovery adds 7
components and picks up 4 previously missed lesions.

---

## Verdict

### H2 — "reasoning detects failures better than uncertainty alone"

**Not supported on either task against the corrected baseline.**

On predicted components the conclusion is unchanged: confidence alone reaches
0.7863 on test, everything reaches 0.8010, and the paired bootstrap on that
difference straddles zero ([−0.0089, +0.0371]). Reasoning features alone are
worse (0.7394).

The recovery result does **not** survive the baseline correction, and this is a
retraction. Against the inflated M0 it looked like the clearest finding in the
phase: confidence 0.5900, reasoning 0.7163, everything 0.7374, with a bootstrap
gap of +0.147 whose CI excluded zero. Against the corrected M0 those become
0.4565, 0.5548 and 0.5325, with a gap of +0.076 whose **CI straddles zero**.

The reason is that the fusion rule was manufacturing the candidates. Averaging
probabilities inflated the sub-threshold region, giving 872 recovery candidates
with 76 positives on test; with logit fusion there are 229 with 27. Validation
has 12 positives in total, and the cross-validated AUROC of the full detector
there is 0.4434 ± 0.2607 — indistinguishable from noise. There is no longer
enough signal, or enough data, to claim anything.

The hand-written constraint set remains the weakest detector on both tasks.

**The asymmetry argument still stands as a hypothesis** — above threshold the
model has committed to a view, below threshold it has not — but it is now
untested rather than supported. Part 2's cross-validation gives ~738 out-of-fold
training cases instead of 76 validation cases, which is the right basis to
decide it on.

### H3 — "refinement improves the prediction"

**Not supported as a Dice improvement. Supported as a moderate false-positive
reduction, at a real cost in lesion sensitivity.**

Against the corrected baseline the best variant raises mean Dice by 0.0134 with
a CI from −0.0067 to +0.0356, improves 50 cases and degrades 27, cuts
false-positive volume 27% (15.96 → 11.71 mL), clears 15 of 90 tumour-free
patients, and improves HD95 by 19.9 mm and ASSD from 43.8 to 32.6 mm. Unlike
the weak-baseline version, every distance metric now moves the right way at
once, because the retuned threshold removes far less.

The cost is 60 true lesions, 27 of them ≥ 1 mL — the size class carrying 90% of
missed volume in Phase II — and lesion sensitivity falling from 0.552 to 0.518.
That is less than half the damage the weak baseline implied, but it is still
real, and the Dice gain that pays for it is not statistically distinguishable
from zero.

Action B remains nearly inert: at the conservative point it recovers 7 lesions
across 200 patients and moves Dice by −0.0015. Reflection is still
indistinguishable from A+B, because Q is built from the detector that chose the
actions.

**What this says for the next step.** The honest position is that the detector
ranks well (AUROC 0.80) but no fixed threshold has yet been shown to deliver a
Dice gain, and the only operating points that cut false positives meaningfully
also delete large lesions. Two things would change that, and Part 2 provides
both: a detector fitted on ~738 out-of-fold training cases rather than 76
validation cases, and an operating point chosen to hold lesion sensitivity
within a point of M0 rather than to maximise Dice, which is indifferent between
a missed cancer and a spurious blob.
