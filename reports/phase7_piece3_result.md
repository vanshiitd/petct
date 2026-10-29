# Piece 3 result: class-balanced sample weighting -- falsified

Confirmatory case-level evaluation, `scripts/phase7_piece3_eval.py` ->
`D:\data\reports\phase7_piece3_eval.json`, same recipe as pieces 1/2.

| policy (cross-fitted, OOF, T=10, p_failure>=0.90) | dDice | tumour-free FP cleared |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 230/385 |
| **(b) hand-built policy** | **+0.0065 [+0.0025, +0.0118]** | 231/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 237/385 |
| (d) shrinkage policy (piece 1, falsified) | +0.0029 [+0.0015, +0.0048] | 50/385 |
| (e) pairwise ranking (piece 2) | +0.0033 [-0.0035, +0.0099] | 59/385 |
| (f) **class-balanced ranking (piece 3)** | +0.0033 [-0.0035, +0.0100] | **59/385** |

Validation: n=38, dDice +0.0008 [-0.0010, +0.0030] -- identical to piece 2
to four decimal places.

## Verdict: falsified

Piece 3's criterion is specific: "falsified if it does not reduce the
selection's concentration on a single strategy ... without costing the
primary metric." The intervention (per-row sample weight
`1 / count(pred_class)` within training, lambdarank otherwise unchanged)
**did** measure a reduction in the cross-fitted fast-proxy concentration on
`topology_remove`, from 99.8% (piece 2) to 97.9% -- 14 of 677 selected
actions switched from `topology_remove` to `recover`. That is a real,
measured effect, not noise (same inner/outer fold structure, same grid).

But at the case level it changes nothing: the agent run is **identical to
piece 2** in FP-clearance (59/385 both) and within rounding on dDice. 14
reassignments out of 677, concentrated on components where the second-best
option was `recover` (do nothing) rather than a genuinely different
corrective action, does not move Dice or clear a single additional
tumour-free false positive. The intervention is falsified on its practical
purpose even though it is not falsified on its literal, narrowly-worded
measurement.

## Why the weighting was too weak

Per-row inverse-frequency weighting rescales each row's contribution to the
lambdarank gradient, but it does not change *what the training data contains
within each component's candidate group* -- and for the overwhelming
majority of components, `topology_remove` genuinely is the best measured
action among the few candidates generated for it. Reweighting changes how
hard the model tries to get a comparison right; it does not manufacture
comparisons against the minority-class actions that were never generated as
candidates for the majority-class components in the first place. The
training-distribution imbalance piece 3 targeted is real (19957
`spurious_other` actions vs. 148 `spurious_physio`), but it is largely an
imbalance in *which components exist*, not in how much the loss function
weighs the actions it already has -- a resampling of *components*, not of
loss weight, would be needed to move this further, and even that would only
help on components that actually have a competitive alternative candidate.

## Carried forward to piece 4

The joint diagnosis+selection scorer (piece 4) is the architectural change
that can actually address this: instead of ranking candidates that were
generated independently of the scorer, a joint model scoring (act-or-not) x
(which strategy) together has the opportunity to influence which
alternatives get considered in the first place, rather than reweighing a
fixed, already-skewed candidate set after the fact. Piece 3's result argues
that reweighting alone, without changing the candidate generation or the
decision surface, is not sufficient -- which is exactly the limitation piece
4 is pre-registered to test.
