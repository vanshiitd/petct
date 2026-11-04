# Phase XIV: calibrated, safety-guarded context detector -- pre-registered

Written before any Phase XIV model is fitted or any number for it is
computed. Branch `phase14-calibrated`, from `phase13-detector`.

## Disclosure

Phase XIII out-of-fold and validation results were read before this plan
was written, and they motivated it:

- the context detector cleared 116 tumour-free patients against the
  hand-built 58, out-of-fold;
- it failed Dice and lesion non-inferiority, because five patients lost a
  true lesion that the detector called spurious at p >= 0.90;
- it put twice as many components through the 0.90 gate (630 -> 1273).

The guard below was **not** designed from those five patients' images or
feature values. Out-of-fold and validation are therefore not confirmatory.
**Test, still unread in every phase, is.**

## Diagnosis this targets

The Phase XIII detector ranks components well but is **overconfident**: its
0.90 no longer means what the hand-built gate assumed. Phase XIV adds four
independent defences against overconfidence. Training-fold data decides how
much of each to use.

1. **Calibration.** Each class head of both detectors gets isotonic
   regression (`out_of_bounds="clip"`). It is fitted on inner cross-fitted
   predictions only, so a calibrated 0.90 is an empirical 90% on data the
   heads never saw.
2. **Ensembling with the old detector.** The candidate `ens` is the mean of
   the calibrated old and calibrated new detectors, per class. Averaging two
   differently-informed models shrinks confident disagreements.
3. **Agreement guard.** Act only if the old and new detectors' argmax classes
   agree, and the old calibrated max-class probability is >= 0.50.
4. **Solo-component guard.** Never act on a component that is the only
   predicted component in its case.
5. **Gate raised as needed.** The gate `t` is chosen from {0.85, 0.90, 0.95}.

## Configurations

There are 2 x 3 x 4 = 24 configurations:
- detector in {`new_cal`, `ens`};
- gate `t` in {0.85, 0.90, 0.95};
- guard in {none, agree, solo, agree+solo}.

Classes, strategies (`phase4_policy_noban.json`), labels, the `CLF` heads
and the 21 Phase XIII features are unchanged.

## Nesting

For outer fold k in the Phase IIIb folds 3-7, with training folds T:

- **Heads.** Old and new heads are fitted on T and score fold k. This
  reproduces `audit_crossfit.npz` and `phase13_det_oof.npz`, checked exactly.
- **Calibrators.** These are fitted on the inner cross-fitted predictions
  over T and applied to fold k.
- **Selection data.** For selection on T, each inner fold is calibrated by
  calibrators fitted on the other inner folds of T. Selection never sees a
  calibrator fitted on its own rows.

**Selection on T.** The hand-built reference on T is the old *uncalibrated*
detector at 0.90, scored with the same inner predictions. Each component's
measured outcome is the hand-built strategy for that configuration's argmax
class (`d_qdice`, `lesions_lost`, as in Phases XI-XII). A configuration is
**admissible** on T only if its acted set has:

- (a) lesions lost <= the hand-built reference's lesions lost on T;
- (b) summed `d_qdice` on tumour-positive cases >= the hand-built's on T.

These are the two non-inferiority endpoints, enforced in training. Among
admissible configurations, take the largest total summed `d_qdice`. Ties go
to `ens` over `new_cal`, then to the higher `t`, then to the stronger guard.
**If none is admissible, roll back to the hand-built detector and gate for
fold k.**

The production configuration (validation, test) is selected the same way on
all out-of-fold data. Heads and calibrators are fitted on all out-of-fold
rows.

## Deployment

The decision goes into a new `--pfail-table`:
- acted rows carry their configuration's probabilities;
- other rows are zeroed;
- recovery rows are copied as in Phase XIII.

The agent runs with `--min-pfail 1e-6 --policy phase4_policy_noban.json --T 10`
(the Phase XI route, verified exact). For rolled-back folds, the old
uncalibrated detector is used with the 0.90 gate, so those cases reproduce
the hand-built run.

## Endpoints

These are unchanged from Phase XIII. All are paired against `au_{split}_attr_policy`.

There are two superiority routes, so each is tested at a Bonferroni level
of two-sided 0.025.

1. **FP-clearance superiority.** More tumour-free patients cleared, with an
   exact McNemar p < 0.025.
2. **Dice superiority.** Paired dDice > 0, with bootstrap p < 0.025.
3. **Dice non-inferiority.** Paired dDice lower 95% bound >= -0.0025.
4. **Lesion non-inferiority.** Lesion hits retained, lower 95% bound >= -5.
5. **Beats the hand-built policy overall.** (1 or 2) and 3 and 4.

**Validation** is a direction check, required for test:
- cleared >= the hand-built policy's;
- dDice >= -0.0025;
- lesions retained >= -1.

**Test** is read **once**, only if 5 holds out-of-fold and validation passes.
It is read for this configuration and for one hand-built test run
(`phase11_eag_hb_test.npz`). **On test, criterion 5 is the confirmatory
result.**

## Pre-specified overconfidence descriptives (held-out, out-of-fold)

For old raw, new raw, new calibrated and ensemble:
- ECE (10 equal-mass bins) and Brier score of max-class P against "argmax
  class is correct";
- **precision at the gate**: the fraction of components at P >= 0.90 whose
  argmax class is correct;
- the number at the gate;
- the fraction of tumour-positive patients' true-lesion components called
  spurious at >= 0.90.

Also reported:
- per-fold selection and the admissibility table;
- the Phase XI proxy subgroup table.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
