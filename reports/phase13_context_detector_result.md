# Phase XIII result: the context detector doubles false-positive clearance, but the pre-registered win fails on safety

Scripts:
- `scripts/phase13_context_features.py`
- `scripts/phase13_detector.py`
- `scripts/phase13_eval.py`

Outputs, written to `D:\data\reports\`:
- `cand_cv_{oof,val,test}_ctx.csv`
- `phase13_det_{oof,val,test}.npz`
- `phase13_det.json`
- `phase13_eval_{oof,val}.json`
- `p13_{oof,val}_det_*`

Pre-registered in `reports/phase13_context_detector_plan.md` (commit d5b6e99).

## Checks that the machinery is honest

- **Component identity.** Every one of the 6462 / 627 / 1508 components
  matches the candidate table voxel for voxel.
- **Out-of-fold route.** The old features reproduce `audit_crossfit.npz`
  exactly: max |dP| = 0.
- **Validation route.** The old features, fitted on all out-of-fold rows,
  reproduce the deployed detector exactly: max |dP| = 0.

The policy, gate, strategies and agent are unchanged. Every difference below
comes from the 21 new features.

## Detection

AUROC, old -> new. Classes not shown change by < 0.03.

| class | out-of-fold | validation |
|---|---|---|
| `spurious_other` (n+ 2490 / 151) | 0.818 -> **0.788** | 0.881 -> **0.932** |
| `fragmentation` | 0.715 -> 0.780 | 0.692 -> 0.858 |
| `under_seg` | 0.793 -> 0.755 | 0.773 -> 0.802 |
| `over_seg` | 0.722 -> 0.751 | 0.849 -> 0.885 |
| components at the 0.90 gate | **630 -> 1273** | 26 -> 44 |

The new detector is far more *confident*: twice as many components clear
the gate. Its ranking of `spurious_other` improves on validation but worsens
pooled out-of-fold.

Permutation importance on that head (AUROC drop) ranks the groups:

| group | AUROC drop |
|---|---|
| within-case rank | 0.013 |
| contrast | 0.012 |
| neighbourhood | 0.009 |
| texture | 0.004 |
| symmetry | 0.003 |
| model profile | 0.001 |

## Pre-registered endpoints

Every row is paired against the hand-built run.

| | out-of-fold (353 positive / 385 tumour-free) | validation (38 / 38) |
|---|---|---|
| tumour-free cleared | **116 vs 58**. Discordant 61 vs 3, McNemar p = 5e-15 -> pass | **8 vs 2**. Discordant 6 vs 0, p = 0.031 -> pass |
| dDice | -0.0043 [-0.0132, +0.0026], p = 0.25 -> superiority fail, **NI fail** | +0.0023 [-0.0000, +0.0068] -> pass |
| lesion hits retained | -2 [-13, +9] -> **NI fail** (needs >= -5) | +0 [+0, +0] -> pass |
| beats the hand-built policy overall | **no** | direction check passes |

**Verdict: falsified on the pre-registered criteria.** Out-of-fold, criterion
1 passes overwhelmingly, but criteria 3 and 4 fail. Under the plan's rule,
**test stays untouched**, even though validation passed.

## Why it fails (descriptive)

The new detector clears false positives from **twice as many tumour-free
patients** as the hand-built policy (116 against 58), with no new false
positive. On tumour-positive patients, 38 are better and 20 worse.

The Dice loss is concentrated. Five patients account for -0.0074 of the
-0.0043 net:

| patient | Dice change | lesion hits | M0 Dice |
|---|---|---|---|
| PETCT_b510436d83_20031114 | -0.91 | -1 (its only lesion) | 0.91 |
| PETCT_245182006a | -0.60 | -1 (its only lesion) | |
| PETCT_a7b323b3fd | -0.48 | -1 (its only lesion) | |
| PETCT_22d07bdc49 | -0.41 | 0 | |
| PETCT_f2f28337ba | -0.23 | -1 | |

In each, a true lesion was rated spurious at p >= 0.90 and deleted. The new
detector is better at telling false positives apart, but it is
**overconfident**: its 0.90 no longer means what the old detector's 0.90 did.
That is a calibration failure, not a ranking failure.

PETCT_22d07bdc49, the case that broke every learned policy from Phase VII on,
is still lost here.

## What this says

For the first time, a change makes a large, unambiguous gain over the
hand-built policy on its own terms: twice the clean tumour-free scans, on
both out-of-fold and validation. On validation it gives up nothing.

What stops it from being a win is a handful of single-lesion patients where
the more confident detector deletes the only lesion. The fix the data points
to is **calibration and a lesion-safety guard**, not more features. Any such
fix is new, has to be pre-registered on its own, and was motivated by these
out-of-fold results.

## Deviations

- The plan text said "22" new features. It was corrected to 21 before any
  number was computed (commit 36428ad).
- Validation was read before out-of-fold had finished. That is permitted:
  validation is a direction check, and nothing was refitted after reading it.
