# Phase X result: the lesion-safety veto cannot be certified -- falsified

`scripts/phase10_veto_select.py` -> `D:\data\reports\phase10_veto.json`.
Pre-registered in `reports/phase10_lesion_veto_plan.md`.

## Pre-registered result

Nested selection (cost ceiling 0.001 Dice per tumour-positive case, no
tumour-free clearance given up):

| outer fold | selected on training folds | held-out lesions saved |
|---|---|---|
| 3 | nothing passes -> no veto | 0 |
| 4 | edge > 30, case p_failure < 0.3, vol < 20 | 0 of 0 |
| 5 | edge > 30, case p_failure < 0.3, vol < 1 | 1 of 1 |
| 6 | nothing passes -> no veto | 0 |
| 7 | nothing passes -> no veto | 0 |

Cross-fitted: **1 of 41** lesion deletions prevented. On all out-of-fold
data no configuration passes either. The veto is not certifiable under the
pre-registered criteria, so no agent was run and **test stays untouched**.

The exploration that motivated the rule estimated a cost of about 0.0008
per tumour-positive case. Under nested selection the training-fold
estimates are 0.0012-0.0015, over the ceiling: lesion deletions are
concentrated in a few folds, and the cost of vetoing the interior false
positives that sit beside them does not average away within a fold. This is
exactly why the plan treated the exploration as non-confirmatory.

## Labelled post-hoc sensitivity (not pre-registered, not confirmatory)

The same nested procedure with the ceiling raised to 0.0015 -- the exchange
rate the current Q-gated final agent already accepts:

| | lesions saved | Dice cost per tumour-positive case | tumour-free clearances lost |
|---|---|---|---|
| veto, ceiling 0.0015 (held-out, component-level) | 25 of 41 | 0.0018 | 0 |
| Q gate, current final agent (agent run, OOF) | 20 of 24 | 0.0015 | 8 |

Held-out cost (0.0018) exceeds the ceiling it was selected under (0.0015) --
the training folds under-estimate it again. The veto keeps every tumour-free
clearance, which the Q gate does not, but costs slightly more Dice per
lesion saved. **It is a different point on roughly the same trade-off
curve, not an improvement on it**, and the component-level numbers have not
been confirmed by an agent run. No claim is made from this table.
