# Phase VIII result: causal LCB-gated surgical override -- falsified

Confirmatory case-level evaluation, `scripts/phase8_eval.py` ->
`D:\data\reports\phase8_eval.json`, same recipe as every Phase VII piece
(`run_audit_oof.sh`'s T=10, `min-pfail`=0.90, no Q-gate), plus `--with-roi`
so the two promoted cells' override action (`roi_repredict`) is actually a
live candidate -- `phase4_policy_noban.json`'s hand-built policy never
lists `roi_repredict` for any class, so this flag is a no-op for every
reference row and every other piece.

| policy (cross-fitted, OOF, T=10, p_failure>=0.90) | dDice | tumour-free FP cleared |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 59/385 |
| **(b) hand-built policy** | **+0.0065 [+0.0025, +0.0118]** | 58/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 52/385 |
| (d) shrinkage (piece 1) | +0.0029 [+0.0015, +0.0048] | 50/385 |
| (e) pairwise ranking (piece 2) | +0.0033 [-0.0035, +0.0099] | 59/385 |
| (f) class-balanced ranking (piece 3) | +0.0033 [-0.0035, +0.0100] | 59/385 |
| (g) **causal hybrid override (Phase VIII)** | **+0.0032 [+0.0019, +0.0048]**, p=0.0005 | **46/385** |

Validation: n=38, dDice +0.0013 [+0.0001, +0.0032], p=0.001 -- same
direction, smallest validation interval of any piece this phase.

## Verdict: falsified

Phase VIII's own criterion -- beat piece 3 on the primary metric, with a
safety margin matched or better -- is not met. dDice (+0.0032) is
statistically the most tightly-estimated positive result of any learned
piece in this project (p=0.0005, the narrowest confidence interval,
[+0.0019, +0.0048]) but ties pieces 1-3's territory, well short of (b)'s
+0.0065. FP clearance (46/385) is the second-worst of any piece evaluated
in this phase or Phase VII, behind only piece 4's 24/385 -- worse than
piece 1's 50/385, despite changing only two narrow, individually-verified
cells rather than the whole selection mechanism.

## Why: a real causal effect, on a metric that didn't protect clearance

The mechanism is traceable directly, not inferred. Of 44 `roi_repredict`
actions the hybrid actually took, 43 were on `spurious_other`
(size-bin 2, the large components) -- the one override whose paired LCB
(+0.0030, win rate 60%) cleared the promotion bar on all out-of-fold data.
The `under_seg` override, the more robust of the two cross-fitted (promoted
in 4 of 5 outer folds, smaller but more consistent effect), fired only
once in the whole real agent run -- it rarely gets the chance to compete
for the budget against `topology_remove` on other components in the same
case.

The causal search promoted `roi_repredict` over `topology_remove` for large
`spurious_other` components because it raises mean `d_qdice` with a
provable margin -- but `d_qdice` rewards *improving* a component's overlap
with ground truth, and `roi_repredict` re-predicts the region rather than
deleting it. `topology_remove` directly removes the component, which is
what actually clears a tumour-free patient's false positive in the
discrete McNemar sense. A component can score better on `d_qdice` after
`roi_repredict` (partially corrected) while still being present, so the
patient is not counted as cleared. **The promotion criterion checked a
paired win-rate on the proxy it optimised, but never checked whether the
override preserved the clinically-relevant discrete outcome (FP presence)
that the action it replaced was specifically good at.**

This is exactly the gap the shared research plan's "protected-subgroup
regression guard" (Darwin Gödel Machine's regression protection) is meant
to close, and this project's own version of it -- a win-rate threshold on
the same proxy metric being optimised -- was not a strong enough guard,
because it never considered a *different* metric the default action was
protecting. A correct implementation would require the promotion criterion
to check LCB-on-FP-clearance specifically, not just LCB-on-`d_qdice`, for
any cell whose default action is doing double duty (correcting the metric
*and* removing a visible false positive). That is a concrete, well-defined
next step, not pursued further here.

## What this adds to the project's record

The mechanism (discrete, LCB-gated, protected local override, hand-built
default otherwise) is sound and the two promoted cells are genuinely
causal, cross-fitted effects, not noise -- this is not a repeat of piece
1's "shrinkage smoothed away the policy" failure or piece 2/3's "collapsed
to one action" failure. It fails for a third, distinct reason: **the
single metric used to decide whether to promote an override was not the
metric that mattered for the action being replaced.** Five of five Phase
VII pieces and this one piece all fail to beat the hand-built policy
(+0.0065 cross-fitted), each for a different, specific, traceable reason --
none of them a repeat of the others.
