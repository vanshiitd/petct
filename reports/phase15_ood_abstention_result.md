# Phase XV result (terminal): out-of-distribution abstention fixes the overconfidence; the pre-registered win is not reached and production rolls back

Script: `scripts/phase15_ood.py`. Evaluation: `scripts/phase13_eval.py --run
p15_oof_ood --tag phase15`.

Outputs, written to `D:\data\reports\`:
- `phase15_ood.json`
- `phase15_ood_{oof,val,test}.npz`
- `phase15_eval_oof.json`
- `p15_oof_ood_*`

Pre-registered in `reports/phase15_ood_abstention_plan.md` (commit d24a32b).

## Answer to the thesis question

**No learned detector or policy was shown to beat the hand-built policy.**
This is the terminal phase, so the question is closed.

The production selection rolled back. Its validation and test tables are
identical to the hand-built tables: max |dP| = 0, and the recovery rows
are identical too. A test run would therefore compare the hand-built policy
with itself, and the difference is zero by construction.

Test was **not** run. It remains unread for every policy and detector from
Phase VII on.

## Checks

- The heads reproduce `audit_crossfit.npz` and `phase13_det_oof.npz` exactly.
- The validation and test tables equal the hand-built tables exactly.

## Out-of-distribution rule

The fence is Q3 + 3 IQR of predicted components per case:
- 34 on folds 3, 5, 6, 7 and in production;
- 30 on fold 4.

About 4-6% of cases abstain:
- 38 out-of-fold cases, PETCT_1285b86bea among them;
- 4 of 76 validation cases;
- 11 of 200 test cases (a count only; no test outcome was read).

**It removes the overconfidence.** On in-support cases, held-out:

| detector | ECE | precision at the 0.90 gate | true lesions called spurious at the gate |
|---|---|---|---|
| old raw | 0.023 | 0.976 | 0.60% |
| new raw | 0.026 | 0.975 | 1.04% |
| new, isotonic | 0.019 | 0.964 | 1.37% |
| ensemble | **0.014** | **0.980** | **0.55%** |

In Phase XIV, including the outlier, the new detector's precision at the
gate was 0.624 and its ECE 0.099.

## Selection

| outer fold | admissible | chosen |
|---|---|---|
| 3 | 6 / 24 | `new_cal`, 0.95, agree |
| 4 | 14 / 24 | `new_cal`, 0.95, none |
| 5 | 8 / 24 | `new_cal`, 0.95, none |
| 6 | 10 / 24 | `ens`, 0.90, none |
| 7 | 0 / 24 | rollback |
| **production** | **0 / 24** | **rollback** |

In Phase XIV, 4 of 5 folds rolled back; with the rule, 1 does.

In production, the nearest configuration (`new_cal`, 0.95, none) has:
- summed `d_qdice` +40.2 against the hand-built +27.9;
- tumour-positive `d_qdice` +2.24 against +1.93;
- but **43 lesions lost against 41**, at the component level.

It misses the pre-registered admissibility rule by two lesions, so the rule
rolls back.

## Pre-registered endpoints, out-of-fold (descriptive, see the plan section 1)

| criterion | result | |
|---|---|---|
| 1 FP superiority | **77 vs 58** cleared; discordant 22 vs 3, McNemar p = 0.0002 | pass |
| 2 Dice superiority | +0.0001, p = 0.80 | fail |
| 3 Dice NI (lower >= -0.0025) | +0.0001 [**-0.0033**, +0.0035] | **fail** |
| 4 lesion NI (lower >= -5) | +0 [-3, +3] | pass |
| 5 beats the hand-built policy | | **no** |

The Dice lower bound misses because of PETCT_22d07bdc49. That patient
contributes -0.0012 and is the same interior false positive that defeated
Phases VII-XII.

## The arc of Phases XIII-XV

| phase | cleared (HB 58) | dDice vs HB [95%] | lesions vs HB |
|---|---|---|---|
| XIII context detector | 116 | -0.0043 [-0.0132, +0.0026] | -2 [-13, +9] |
| XIV + calibration and guards | 71 | +0.0003 [-0.0047, +0.0042] | +5 [-1, +16] |
| XV + OOD abstention | 77 | +0.0001 [-0.0033, +0.0035] | +0 [-3, +3] |

Each defence narrowed the risk interval while keeping a clear,
statistically strong gain in clean tumour-free scans. None of them closed
the last ~0.001 of Dice margin, which a single patient decides each time.

## What this says, for the thesis

1. **Trust machinery works as intended.** The system refused to deploy a
   change it could not certify. It refused even when the point estimates
   favoured it. That is the behaviour a clinical tool should have.
2. **"Know when you don't know" was the decisive defence against
   overconfidence.** With one out-of-support patient removed, calibration
   error halved and precision at the gate returned to 0.98.
3. **The ceiling is per-patient detector error**, not policy design:
   PETCT_22d07bdc49, PETCT_245182006a and PETCT_14f931f634. Raising it needs
   new information, such as better segmentation models or more training
   data. More policy search on the same data does not raise it.

## Deviations

- **Test not run.** Section 4 of the plan says a full rollback is
  "reported as no learned change deployed". Because the deployed tables are
  identical to the hand-built tables, a test run would add no information,
  so it was not run. Test stays unread for this question.
