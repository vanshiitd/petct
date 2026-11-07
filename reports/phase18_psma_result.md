# Phase XVIII result: PSMA external confirmation

**Locked run.** Plan `reports/phase18_psma_plan.md`, lock
`reports/phase18_psma_lock.md`, both arms run once after it. The lock's
deviations section records two aborted starts (patient-id lookup). Both
failed before any output, so no outcome was seen before this run.

## Primary endpoints (pre-registered)

| endpoint | criterion | result | met |
|---|---|---|---|
| **S** safety | exact binomial, H0: R > 0.01, α = 0.05, n = 377 | **0 / 377 harmed**, p = 0.0226; exact 95% UB on R = **0.0079** | **yes** |
| **D** Dice NI | paired dDice lower bound >= -0.0025 | +0.0000 [+0.0000, +0.0000] | **yes** |
| **L** lesion NI | lesion-hit lower bound >= -5 | +0 [+0, +0] | **yes** |

**Confirmed safe transfer (S ∧ D ∧ L): met.**

S needed k = 0, as the plan stated in advance, and that is what happened.

## Secondary endpoints

| endpoint | result | met |
|---|---|---|
| 1 FP-clearance superiority | 0 vs 0 of 19 tumour-free cleared, McNemar p = 1 | no |
| 2 Dice superiority | dDice +0.0000, p = 0.102 | no |
| 5 beats hand-built | -- | no |

P17 does **not** improve on the hand-built policy on PSMA.

## Descriptive

**Base model (FDG nnU-Net ensemble, zero-shot).** It fails badly on PSMA:

| | Dice | FP mL/case | FN mL/case | HD95 | lesion hits |
|---|---|---|---|---|---|
| M0 | 0.1045 | 732.2 | 45.39 | 198.1 | 6003 |
| HB | 0.1057 | 719.2 | 45.38 | 198.0 | 6005 |
| P17 | 0.1057 | 718.9 | 45.38 | 198.1 | 6005 |

The FP volume is consistent with PSMA physiological uptake (salivary glands,
kidneys, bladder) being segmented as tumour. This is a segmenter problem, not
an agent problem, and it is outside the frozen agent's scope.

**Both refinement arms vs M0.**
- dDice +0.0013 [+0.0006, +0.0022];
- +2 lesion hits;
- FP volume -13 mL/case.

**What P17 did.**
- It kept all 189 HB actions and added 7 components, changing 7 cases.
- Attempts 199 vs 192; accepted 172 vs 165.
- It abstained (out of fence, > 34 components) on 67 of 376 cases.

**Fairness subgroups.** There was 0 harm in every subgroup. P17 - HB dDice is
between +0.00000 and +0.00001 throughout.

| subgroup | n | M0 Dice | HB - M0 |
|---|---|---|---|
| 18F | 213 | 0.111 | +0.0002 |
| 68Ga | 164 | 0.109 | +0.0027 |
| GE | 157 | 0.056 | +0.0012 |
| Siemens | 220 | 0.147 | +0.0014 |
| age low | 133 | 0.100 | +0.0020 |
| age mid | 135 | 0.102 | +0.0006 |
| age high | 109 | 0.131 | +0.0012 |

The base model's GE/Siemens gap (0.056 vs 0.147) is the largest disparity.
Both refinement arms leave it unchanged.

## Reading

**What this supports.** Under a large, unplanned shift (new tracer, new
scanners, a base model at Dice 0.10), the certified update:
- harmed no patient (95% UB 0.79%);
- was non-inferior to the hand-built policy;
- abstained heavily where its fence said the case was out of distribution.

That is the safety property the certificate is meant to carry.

**What it does not support.** It is not evidence of benefit. P17 changed 7 of
377 scans and the gains are zero to 4 decimals. Safety here is largely
conservatism: the 0.98 gate and the fence leave the agent nearly inert off
distribution. That is the designed behaviour, but it should be presented as
such, not as an improvement.

## Files

`D:\data\reports\`:
- `p18_psma_{hb,ltt}_*`
- `phase18_agent_{hb,ltt}.log`
- `phase18_eval_psma.json`
- `phase18_eval_psma_safety.json`
- `phase18_psma_tables.json`
