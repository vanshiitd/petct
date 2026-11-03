# Piece 2 result: pairwise ranking + case-level credit assignment

Confirmatory case-level evaluation, `scripts/phase7_piece2_eval.py` ->
`D:\data\reports\phase7_piece2_eval.json`, exact `run_audit_oof.sh` recipe
(T=10, `min-pfail`=0.90, no Q-gate).

| policy (cross-fitted, OOF, T=10, p_failure>=0.90) | dDice | tumour-free FP cleared |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 230/385 |
| **(b) hand-built policy** | **+0.0065 [+0.0025, +0.0118]** | 231/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 237/385 |
| (d) shrinkage policy (piece 1, falsified) | +0.0029 [+0.0015, +0.0048] | 50/385 |
| (e) **pairwise ranking (piece 2)** | **+0.0033 [-0.0035, +0.0099]** | **59/385** |

Validation (check only): n=38, dDice +0.0008 [-0.0010, +0.0030], p=0.421 --
straddles zero, consistent with OOF.

## Piece 2's own pre-registered criterion: not falsified, but barely clears it

Cross-fitted pooled Spearman **0.397**, up from the existing model's 0.31 --
the first condition is met. The fast proxy and the case-level primary metric
both beat piece 1 (+0.0033 > +0.0029 OOF; 59/385 FP cleared vs 50/385) -- the
second condition is met, narrowly. By the letter of the pre-registered
criterion ("falsified if Spearman does not improve *and* the primary metric
does not improve over piece 1"), piece 2 survives.

## It does not clear the phase-level bar (b), and the reason is informative

(e)'s point estimate and interval are, to four decimal places, the same as
(a)'s -- not a coincidence. Checking the experience log directly:

```
topology_remove   606 / 607 accepted actions
suv41               1 / 607
```

**The pairwise ranker collapsed onto a single action for all but one gated
component**, converging to something indistinguishable from "always pick
`topology_remove`" -- 736 of 738 cases have the identical per-case Dice to
the `attr_fixed` run that does exactly that. Fixing the regression-on-noise
problem (reasons 1-2: cross-fitted Spearman up from 0.31 to 0.40) did **not**
fix the collapse (reason 3). This is the expected failure mode given how
`lambdarank` was trained here: `topology_remove` is the correct action for
the large majority of gated components (the `spurious_*` classes vastly
outnumber `boundary`/`fragmentation`/`under_seg` in the training
distribution), so a model optimising aggregate ranking quality learns to bet
on the majority class's preferred action almost everywhere, including on the
minority classes where a different action wins by a wide margin -- exactly
where the hand-built policy's per-class rule pays off and where piece 1's
shrinkage also lost ground.

## Verdict

Piece 2 is **not falsified by its own criterion** (Spearman and the primary
metric both improved over piece 1) but **does not beat the hand-built policy
(b)** and does not yet justify adopting a learned component over it. The
specific mechanism -- training-distribution imbalance across strategies
collapsing the ranker onto the majority action -- is exactly reason 3 from
the retrospective, and is precisely what piece 3 (rebalanced / stratified
training) is pre-registered to address next. Moving to piece 3.
