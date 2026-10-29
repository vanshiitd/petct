# Phase VII analysis plan (pre-registered)

Written and committed **before any Phase VII number is computed**. Branch
`phase7-selfimprove`, from `main` (post-audit).

## Why this phase exists

The audit (`docs/results/audit_reanalysis.md`) established, cross-fitted, that
the hand-built policy beat the learned action-value selector at every budget
tested (T=3 and T=10), and that a flat fixed strategy was the *weakest* of the
three. The project's own retrospective (this conversation) named six concrete,
falsifiable reasons for that:

1. effect sizes are tiny relative to label noise (median action ≈ 0.0002)
2. the learned model's own ranking quality was weak (cross-fitted Spearman ≈ 0.31)
3. the learned model collapsed toward one dominant action
4. the hand-built policy is close to a strong, low-variance baseline for this
   problem (a per-class empirical mean)
5. more training data did not improve the selector (flat self-improvement curve)
6. the `p_failure` gate already filters out most of the decision surface,
   leaving little for a learned selector to add

Five interventions are pre-registered here, one per reason (reason 2 and 3 are
addressed by one intervention). They are implemented and evaluated **in order**,
each as its own commit, each compared against the existing cross-fitted
reference points from the audit:

| reference (cross-fitted, OOF, T=10, p_failure>=0.90) | dDice | tumour-free FP cleared |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 230/385 |
| (b) hand-built policy | **+0.0065 [+0.0025, +0.0118]** | 231/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 237/385 |

(b) is the bar every new policy must clear to be worth adopting.

## Non-negotiable discipline, carried over from the audit

- **Every new component is fit on the Phase IIIb patient-grouped folds
  (3, 4, 5, 6, 7), and evaluated on the fold that did not see it** -
  i.e. genuinely cross-fitted, never in-sample. This includes any
  hyperparameter (shrinkage strength, model depth, learning rate): it is
  selected by an **inner** cross-validation inside the four training folds of
  each outer fold, never by looking at the held-out fold, validation, or test.
- **Validation is a check, never a fitting set.** Test is read once, at the
  very end of the whole phase (after all five pieces), and only for the
  configuration(s) that survive out-of-fold and validation. Every number
  before that point is out-of-fold or validation and is reported as such.
- **No existing report, figure, JSON, prediction, or script from a prior phase
  is modified.** Phase VII writes new files only.

## Endpoints

**Two distinct metrics, not one, and they must not be conflated** - an error
made and caught in drafting this plan, recorded here rather than silently
fixed:

- **Fast proxy (component-level):** mean `d_qdice` of the strategy selected for
  each gated component, out-of-fold, cross-fitted. Cheap (no agent run, pure
  table lookup), used to screen a candidate policy before spending compute on
  a full run.
- **Confirmatory (case-level):** mean Dice over tumour-positive cases, before
  vs. after running the *actual* `phase5_agent.py` loop with the candidate
  policy wired in via `--avpkl`, T=10, `min-pfail`=0.90, no Q gate - the exact
  recipe `run_audit_oof.sh` used for rows (a)/(b)/(c) above, paired bootstrap
  over patients. **This is the only metric directly comparable to
  +0.0033 / +0.0065 / +0.0052**, and it is the one that decides whether a piece
  passes or fails its falsification criterion.

The fast proxy is informative but is not evaluated on the same quantity (it
averages over every gated component, including those on tumour-free cases
scored by the `d_qdice` convention) and must never be reported next to the
case-level reference numbers as if directly comparable.

Also tracked at case level throughout: tumour-free patients with any false
positive (exact McNemar), lesion sensitivity (safety margin -0.01, binding).

## The five pieces, in the order they will be built

1. **Hierarchical shrinkage policy** (addresses reason 4). Empirical-Bayes
   partial pooling over (failure class -> class x size-bin -> class x size-bin x
   SUV-bin), shrinkage strength selected by nested cross-validation on
   out-of-fold alone. Falsified if it does not beat the hand-built policy (b)
   on the primary metric, cross-fitted.
2. **Pairwise ranking objective + case-level credit assignment** (addresses
   reasons 1-2). Replaces the regression-on-raw-delta target with a
   same-component pairwise comparison objective. Falsified if cross-fitted
   Spearman does not improve over the existing 0.31 and the primary metric
   does not improve over piece 1's result.
3. **Rebalanced / stratified training** (addresses reason 3). Re-fits piece 2's
   model on a strategy-stratified sample. Falsified if it does not reduce the
   selection's concentration on a single strategy (measured directly) without
   costing the primary metric.
4. **Joint diagnosis+selection scorer** (addresses reason 6). A single model
   scoring (act-or-not) x (which strategy) jointly, replacing the two-stage
   hard gate + selector. Falsified if it does not beat piece 3 on the primary
   metric at matched or better safety margin.
5. **Closed-loop re-fitting with active learning** (addresses reason 5). The
   only piece that is evaluated over *time* rather than once: simulated
   sequential re-fitting as new verified out-of-fold outcomes accrue, with
   active sampling of under-represented (class, strategy) cells. Falsified if
   the curve does not rise measurably above piece 4's flat/ceiling behaviour.

## Falsification of the phase as a whole

Phase VII is reported as **not successful** if no piece's final configuration
beats the hand-built policy (b) on the primary cross-fitted metric. Partial
results are reported in full regardless of the overall verdict, exactly as
every prior phase in this project has done.
