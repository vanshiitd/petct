# Phase IX result: no gate can be certified -- falsified at selection

`scripts/phase9_class_gates.py` -> `D:\data\reports\phase9_class_gates.json`.
Pre-registered in `reports/phase9_class_gates_plan.md`.

## The in-sample headroom was an artefact

The plan's motivating table used the in-sample p_failure that
`phase5_action_value.build()` produces. Recomputed with the cross-fitted
p_failure and class from `audit_crossfit.npz` -- what the deployed agent
actually sees out-of-fold -- for the hand-built policy's own pick:

| `spurious_other`, p_failure band | n | mean `d_qdice` | LCB | harmful | harm-rate 95% UB |
|---|---|---|---|---|---|
| [0.90, 1.0] -- **deployed today** | 590 | +0.0456 | +0.0295 | 38 (6.4%) | 0.084 |
| [0.85, 0.90) | 152 | +0.0214 | -0.0035 | 36 (24%) | 0.300 |
| [0.80, 0.85) | 179 | +0.0081 | +0.0005 | 43 (24%) | 0.299 |
| [0.75, 0.80) | 231 | +0.0261 | +0.0054 | 62 (27%) | 0.321 |
| [0.70, 0.75) | 248 | +0.0182 | +0.0028 | 75 (30%) | 0.354 |
| [0.65, 0.70) | 260 | +0.0112 | -0.0004 | 84 (32%) | 0.374 |
| [0.60, 0.65) | 276 | +0.0178 | +0.0012 | 98 (36%) | 0.405 |
| [0.50, 0.60) | 565 | +0.0062 | -0.0022 | 209 (37%) | 0.405 |

(harm = `lesions_lost > 0`, or a tumour-positive case with `d_dice < -0.01`.)

In-sample, the [0.80, 0.90) band showed **zero** harmful outcomes; cross-fitted
it is **24%**. This is the audit's in-sample-vs-cross-fitted finding (R1(a))
again, at the component level, and it is why the plan required every number
to be recomputed cross-fitted before anything was promoted.

## Verdict: falsified, robustly

No slice of any class passes the pre-registered safety bound (harm-rate 95%
UB <= 2%) in any of the five outer folds, so no gate is relaxed and the
gated policy is the hand-built policy exactly. No agent run was needed --
with every gate at 0.90 the wrapper is behaviourally identical by
construction.

The verdict does not depend on the 2% threshold. **Labelled sensitivity
check, not pre-registered:** even a ceiling as permissive as the deployed
region's own upper bound (8.4%) admits nothing, because every slice below
0.90 has an upper bound of at least 30%. The gap is not marginal.

Several slices do have a positive benefit LCB -- clearing false positives
pays on average even at p_failure 0.6 -- which is exactly why piece 4's
global gate removal looked partly promising on its proxy and then cost
lesion sensitivity in the real run. The benefit is bought with lesions at a
rate of one action in four. **The 0.90 gate is not conservative scaffolding:
below it the detector's "spurious" label is wrong about a quarter of the
time, and wrong in the direction that deletes tumour.**

## What this leaves

Inside the deployed region the harm rate is 6.4% (38 of 590 actions delete
a real lesion), and those actions are the source of the agent's measured
lesion-sensitivity cost (-0.0034). That is the one place left where a
change can make the agent both safer and better without touching what it
already does well -- a veto on the specific removals that delete tumour.
