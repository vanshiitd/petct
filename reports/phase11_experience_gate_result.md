# Phase XI result: the experience-trained gate clears far more false positives -- but the pre-registered win fails

Scripts: `scripts/phase11_experience_gate.py`, `scripts/phase11_eval.py`.
Outputs, written to `D:\data\reports\`: `phase11_eag.json`,
`phase11_eval_{oof,val}.json` and `p11_{oof,val}_eag_*`.
Pre-registered in `reports/phase11_experience_gate_plan.md` (commit 791c02e).

## Checks that the machinery is honest

- **Nested stage 1.** It reproduces `audit_crossfit.npz` exactly: maximum
  |dP| = 0 in every outer fold.
- **Deployment route.** The table at the hand-built point reproduces the
  hand-built runs case for case. Out-of-fold, 10 of 10 smoke cases match
  `au_oof_attr_policy`. On validation, all 76 cases match `au_val_attr_policy`.
  Any difference below is therefore the gate's doing.

## Selection

On three of the five outer folds the training-fold selection chose
`g = 0.50, h = 0.02`: act down to `p_failure` 0.5, but only where predicted
harm is <= 2%. On folds 3 and 7 nothing met the "halve the lesions lost"
constraint, so those folds fell back to the hand-built point. The production
choice, made on all out-of-fold data, is `g = 0.50, h = 0.02`.

## Pre-registered endpoints

Every row is paired against the hand-built policy's own run.

| | out-of-fold (353 positive / 385 tumour-free) | validation (38 / 38) |
|---|---|---|
| 1 lesion hits retained, EAG - HB | +1 [-3, +4] -> **fail** | -1 [-6, +3] -> **fail** |
| 2 dDice, EAG - HB (NI margin -0.0025) | -0.0017 [-0.0046, -0.0001] -> **fail** | -0.0025 [-0.0077, +0.0008] -> **fail** |
| 3 tumour-free cleared (NI, slack 2) | **75** vs 58 -> pass | **8** vs 2 -> pass |
| 4 win | **no** | **no** (direction check fails) |

| vs M0 | dDice | lesions lost | tumour-free cleared |
|---|---|---|---|
| hand-built policy | +0.0065 [+0.0025, +0.0118] | 24 | 58 / 385 |
| final agent (Q gate) | +0.0050 [+0.0015, +0.0097] | 4 | 50 / 385 |
| **experience-trained gate** | +0.0047 [+0.0018, +0.0093] | 23 | **75 / 385** |

**Verdict: falsified on the pre-registered criteria.** The gate is not safer
than the hand-built policy: it loses 23 lesions against 24. The Dice
non-inferiority bound is missed. Validation fails its direction check. Under
the plan's rule, **test stays untouched.**

## What it does do (labelled: not a pre-registered superiority test)

1. **It clears false positives from 30% more tumour-free patients, with
   none made worse.** Paired out-of-fold, the gate cleared 20 patients the
   hand-built policy did not, and the hand-built policy cleared 3 the gate did
   not. The exact McNemar p is 0.0005. On validation the gate cleared 8
   patients against 2. In both splits it added no new false positive. This is
   the most false-positive clearance of any agent in the project.
2. **Its Dice cost is one patient.** PETCT_22d07bdc49 alone accounts for
   -0.0013 of the -0.0017 out-of-fold, and PETCT_14f931f634 for -0.0021 of
   the -0.0025 on validation. PETCT_22d07bdc49 is the large interior false
   positive that every learned policy in Phases VII-X also failed to delete.
3. **No proxy subgroup is harmed.** In every lesion-count and M0-Dice
   tertile, and among tumour-free patients, the gate's change versus M0 is
   >= 0 (out-of-fold, no upper bound < 0). The gain is largest where M0 is
   weakest: +0.0112 in the low-Dice tertile.
4. **Its harm is certified and lower.** In the acted set, held-out, the
   measured harm rate is 5.6% (40 of 709), with a 95% upper bound of 7.3%. For
   the hand-built policy it is 6.7% (42 of 628), with a 95% upper bound of 8.6%.
   The gate acts more often, and a smaller fraction of its actions harm.
5. **It is calibrated.** These are cross-fitted figures:

   | model | ECE | prevalence |
   |---|---|---|
   | harm model | 0.047 | 0.23 |
   | `spurious_other` head | 0.050 | 0.39 |
   | every other class head | < 0.034 | — |

## What this says

The gate moves the agent to a different point on the trade-off, not past it.
It turns some of the hand-built policy's Dice gain on tumour-positive
patients into a large, statistically clear gain in clean tumour-free scans.
Lesions lost stay the same. The policy-level pattern of Phases VII-X holds a
fifth time: the remaining deficit sits on one or two patients whose large
blob the detector misreads. Only a better per-component detector can reach
them.

## Deviations

None. Validation was read before out-of-fold had finished. That is permitted:
validation is a direction check, and nothing was refitted after reading it.
