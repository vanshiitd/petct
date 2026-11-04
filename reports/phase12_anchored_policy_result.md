# Phase XII result: the anchored policy never does worse than the hand-built policy out-of-fold, but it does not beat it

Scripts: `scripts/phase12_anchored.py`, `scripts/phase12_eval.py`.
Outputs, written to `D:\data\reports\`: `phase12_anchored.json`,
`phase12_eval_{oof,val}.json` and `p12_{oof,val}_anchored_*`.
Pre-registered in `reports/phase12_anchored_policy_plan.md` (commit 05523c4).

## Verdict

**Not achieved on the pre-registered criteria.**
- Out-of-fold, criterion 1 (FP-clearance superiority) fails.
- Validation fails its direction check on lesions.
- Under the plan's rule, **test stays untouched**.

## Selection: the promotion controller mostly rolls back

On each outer fold, a configuration could be promoted only if its added
actions met all of these on the training folds:
- causal-utility LCB > 0;
- Clopper-Pearson harm upper bound <= 5%;
- at most 2 lesions lost;
- no regression on tumour-positive patients.

| outer fold | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| selection | rollback | rollback | rollback | g 0.60, h 0.01 | rollback |

Many configurations had a clearly positive utility LCB. They failed on the
harm bound or on lesions lost, which is what the gate is for.

The production configuration, chosen on all out-of-fold data, is **g = 0.60,
h = 0.02**. Its added set has 263 components, utility LCB +0.038, harm
2/263 (upper bound 2.4%) and 2 lesions lost. Applied to validation, it acts
on 42 components; the hand-built policy acts on 26.

## Pre-registered endpoints

All are paired against the hand-built run.

| | out-of-fold (353 positive / 385 tumour-free) | validation (38 / 38) |
|---|---|---|
| tumour-free cleared | 61 vs 58. Discordant 3 vs 0, McNemar p = 0.25 -> **fail** | 7 vs 2. Discordant 5 vs 0, p = 0.0625 -> pass (direction) |
| dDice (NI margin -0.0025) | +0.0000 [-0.0000, +0.0001] -> pass | -0.0020 [-0.0063, +0.0002] -> pass (direction) |
| lesion hits retained | +0 [+0, +0] -> pass | **-2** [-6, 0] -> **fail** (needs >= -1) |
| beats the hand-built policy overall | **no** | direction check **fails** |

**Out-of-fold, the anchored policy weakly dominates the hand-built policy.**
- It clears 3 more tumour-free patients.
- It gives up no Dice and no lesions.
- No proxy subgroup is harmed.

But only one fold promoted anything. The gain is too small to reach
significance.

**Validation shows where the trust region leaks.** All of the lesion deficit
is PETCT_14f931f634. The detector labelled two of its real lesions
`spurious_other` at p_failure 0.72 and 0.79, and the verifier rated them
safe (predicted harm <= 2%). This is the same patient that cost Phase XI its
validation Dice.

## Pre-specified descriptives

These are held-out, component level, summed over the outer folds.

**Experience faithfulness.** The learned verifier is compared with the same
rule using harm scores shuffled among the eligible candidates (20 shuffles).

| verifier | added | summed d_qdice | lesions lost |
|---|---|---|---|
| learned | 27 | +0.01 | **0** |
| shuffled | 27 | +0.36 | 3.75 |

The learned experience matters causally. With the same number of actions,
it prevents every lesion loss the shuffled rule causes. It pays for that
safety by giving up the small Dice gain.

**Self-improvement curve.** Fold 6 only; the others rolled back.

| training patients used | added | summed d_qdice | lesions lost |
|---|---|---|---|
| 10% | 72 | +3.24 | 4 |
| 25% | 62 | +3.05 | 1 |
| 50% | 37 | +0.02 | 0 |
| 100% | 27 | +0.01 | 0 |

With more experience, the verifier gets **more conservative and safer, not
more productive**. It learns which below-gate actions harm and stops taking
them. The Dice gain the small-data verifier "found" came with lesion losses.

**Calibration** of the harm model on below-gate candidates:
- ECE 0.044, Brier 0.162 (n 2238, prevalence 0.27);
- it under-predicts harm by up to 8 points in the 0.05–0.30 range.

## RL-for-LLM / CAVE-PET reading

| element | outcome |
|---|---|
| reference policy + one-sided trust region | worked as designed: the hand-built actions were never lost out-of-fold |
| verifier-screened additions (rejection sampling) | out-of-fold safe. Failed on one validation patient whose error the verifier cannot see |
| promote/rollback controller | correctly refused 4 of 5 folds. This is conservative, auditable self-improvement |
| faithfulness control | positive: the experience is used causally |
| experience curve | improvement means fewer, safer actions; no further Dice gain |

## What this says

This is the sixth policy-level attempt (Phases VII–XII). Each one ends at
the same wall. Every remaining deficit against the hand-built policy comes
from a few patients where the detector is **confidently wrong** about a
component's class. Examples are PETCT_22d07bdc49 and PETCT_14f931f634.

No gate, verifier or controller built on the detector's own outputs can
recover from that. The remaining lever is the per-component detector itself,
retrained or given new evidence.

## Deviations

None. Validation was read before out-of-fold had finished. That is permitted:
validation is a direction check, and nothing was refitted after reading it.
