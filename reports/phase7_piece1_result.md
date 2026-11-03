# Piece 1 result: hierarchical shrinkage policy -- FALSIFIED

Confirmatory case-level evaluation, `scripts/phase7_piece1_eval.py` ->
`D:\data\reports\phase7_piece1_eval.json`, exact `run_audit_oof.sh` recipe
(T=10, `min-pfail`=0.90, no Q-gate), paired bootstrap over patients, 2000
resamples, same `boot_dice`/`mcnemar_fp`/`lesion_sens` functions the audit used.

| policy (cross-fitted, OOF, T=10, p_failure>=0.90) | dDice | tumour-free FP cleared |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 230/385 |
| **(b) hand-built policy** | **+0.0065 [+0.0025, +0.0118]** | 231/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 237/385 |
| (d) **hierarchical shrinkage policy (piece 1)** | **+0.0029 [+0.0015, +0.0048]** | **50/385** |

Validation (check only, not the deciding split): n=38 tumour-positive cases,
dDice +0.0012 [-0.0000, +0.0032], p=0.064 -- straddles zero, consistent with
the OOF verdict.

## Verdict: does not beat (b). Piece 1 is falsified per the pre-registered criterion.

This is the opposite of what the fast proxy predicted (cross-fitted
component-level mean `d_qdice` +0.039, 95.3% selection precision -- see
`phase7_shrinkage.json`), which is exactly why the plan pre-registered the
proxy as non-decisive and required the real agent run before any claim.

## Why the proxy and the case-level result diverge

The shrinkage estimator pulls every thin (class x size-bin x SUV-bin) cell
toward its parent mean. For the cells gated in, the parent mean is dominated
by the large pool of components where the best action is `recover`
(do nothing / revert) -- the safe, low-variance choice. Shrinkage therefore
makes the policy *more* conservative than the hand-built rule exactly where
the hand-built rule's per-class override mattered: it still picks the
population-safe action on many components the hand-built policy would have
corrected. That raises selection precision (the proxy) because "do nothing"
is rarely wrong, but it collapses the operational effect -- 604 accepted
actions / 738 cases (0.82/case) here vs. a much higher action rate under (b),
and tumour-free FP clearance drops from 231/385 to 50/385. A policy that acts
less often cannot clear as many false positives, regardless of how accurate
its proxy score is on the actions it does take.

This is reason 4 from the retrospective, confirmed rather than fixed: the
hand-built policy's per-class hard override is not just "close to" a shrunk
empirical mean, it is doing something a smooth shrinkage estimator trades
away -- committing to the class-specific action even in thin cells, instead
of regressing it toward a population answer that is safe on average but wrong
for the classes that matter (boundary/fragmentation/under-segmentation, where
`volume_grow` beats `recover` by a wide, if noisily-estimated, margin).

## Carried forward to piece 2

Pairwise ranking + case-level credit assignment does not inherit this specific
failure mode (it will not partial-pool in a way that erases a class's
preferred action), but the result argues for keeping an explicit per-class
floor/override on top of whatever score piece 2 produces, rather than trusting
a smooth statistic to preserve rare-but-large per-class effects. Noted for
piece 4 (joint scorer) design, not implemented here -- piece 2 is evaluated
on its own terms first, as pre-registered.
