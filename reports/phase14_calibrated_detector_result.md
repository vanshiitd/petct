# Phase XIV result: safer and cleaner than the hand-built policy out-of-fold, but Dice non-inferiority misses by one patient

Script: `scripts/phase14_calibrated.py`. Evaluation: `scripts/phase13_eval.py --run
p14_oof_cal --tag phase14`.

Outputs, written to `D:\data\reports\`:
- `phase14_cal.json`
- `phase14_cal_{oof,val,test}.npz`
- `phase14_eval_oof.json`
- `p14_oof_cal_*`

Pre-registered in `reports/phase14_calibrated_detector_plan.md` (commit fee8132).

## Checks

The out-of-fold heads reproduce `audit_crossfit.npz` and
`phase13_det_oof.npz` exactly (max |dP| = 0).

## Selection

| outer fold | admissible configs | chosen |
|---|---|---|
| 3 | 9 / 24 | `new_cal`, t = 0.85, agree guard |
| 4 | 0 / 24 | rollback |
| 5 | 0 / 24 | rollback |
| 6 | 0 / 24 | rollback |
| 7 | 0 / 24 | rollback |
| production (all out-of-fold) | 0 / 24 | **rollback** |

Production rolled back, so the validation and test tables are the
hand-built detector at 0.90. The validation run would reproduce the
hand-built run, so it was not repeated. Test is untouched.

## Pre-registered endpoints, out-of-fold

All are paired against `au_oof_attr_policy`. Only fold 3 differs from
hand-built.

| criterion | result | |
|---|---|---|
| 1 FP superiority | **71 vs 58** cleared; discordant 13 vs 0, McNemar p = 0.0002 | pass |
| 2 Dice superiority | +0.0003, p = 0.82 | fail |
| 3 Dice NI (lower >= -0.0025) | +0.0003 [**-0.0047**, +0.0042] | **fail** |
| 4 lesion NI (lower >= -5) | **+5** [-1, +16] | pass |
| 5 beats the hand-built policy | | **no** |

| vs M0 | dDice | lesions lost | tumour-free cleared |
|---|---|---|---|
| hand-built | +0.0065 [+0.0025, +0.0118] | 24 | 58 / 385 |
| **Phase XIV** | **+0.0068** [+0.0005, +0.0133] | **19** | **71 / 385** |

**Verdict: falsified on the pre-registered criteria.** Point for point, the
policy is better than the hand-built policy on every axis:
- more patients cleared;
- 5 fewer lesions lost;
- higher Dice.

But the Dice non-inferiority bound misses. One patient, PETCT_245182006a
(-0.60 Dice, its only lesion deleted), contributes -0.0017 and widens the
bootstrap interval past the margin.

## Overconfidence descriptives

Held-out, max-class P at the 0.90 gate.

| detector | ECE | at gate | precision at gate | true lesions called spurious at gate |
|---|---|---|---|---|
| old raw | 0.061 | 630 | 0.910 | 1.3% |
| new raw | 0.099 | 1273 | 0.624 | 12.2% |
| new, isotonic | 0.113 | 916 | 0.483 | 12.2% |
| old + new ensemble | **0.056** | 378 | **0.905** | **0.9%** |

On validation, all four have precision at the gate of 0.92-0.94.

**The overconfidence is one patient.** PETCT_1285b86bea (fold 7) has 487
predicted components. The largest any training fold has is 139. Its true
lesion count is 1105. The new detector's within-case features extrapolate
there: 452 of its components were called `spurious_other` at >= 0.90, and
all were wrong.

Per fold, the new detector's precision at the gate is:

| fold | old | new |
|---|---|---|
| 3 | 0.89 | 0.92 |
| 4 | 0.99 | 0.98 |
| 5 | 0.97 | 0.97 |
| 6 | 0.97 | 0.97 |
| 7 | 0.74 | **0.27** |

In the agent, the per-case iteration cap held that patient's damage to
-0.001 Dice and -2 lesion hits. In the component-level training statistics,
though, its ~450 wrong calls counted as lost lesions. This had two effects:

- **It contaminated the isotonic calibrators.** That is why calibration made
  precision worse rather than better.
- **It made every configuration inadmissible on any training set that
  contains fold 7**, which is folds 4, 5, 6 and production.

Fold 3's training set contains fold 7 too, but the agreement guard
survived there.

Ensembling with the old detector was the effective overconfidence defence:
- best ECE;
- precision back to 0.905;
- fewer true lesions called spurious than either detector alone.

## What this says

The trust machinery now works the way it should:
- it refused to promote anything it could not certify;
- where it did promote, the result beat the hand-built policy on every point
  estimate.

The remaining obstacles are two single patients:
- an out-of-distribution case that poisons training-fold statistics;
- one single-lesion deletion that widens the Dice interval.

The principled response to the first is **out-of-distribution abstention**.
A case whose component count lies outside the training range is handed to
the hand-built detector, because the model has no evidence there. That is
new and would need its own pre-registration. It is motivated by these
results.

## Deviations

None.
