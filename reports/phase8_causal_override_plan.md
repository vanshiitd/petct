# Phase VIII: causal, LCB-gated surgical overrides -- pre-registered

Written before any number in this document is computed. Branch
`phase8-causal-override`, from `main` (after Phase VII, PR #10, and the
H2/H4 reanalysis, PR #11).

## Why this is different from Phase VII

A reader shared a research plan (CAVE-PET) built around LLM self-improving
agents, with a specific mechanism this project has not tried: **promote a
learned rule only if its causal effect, measured where it actually fires,
clears a lower-confidence-bound margin -- and keep the default everywhere
else.** Every Phase VII piece instead replaced the *entire* selection
mechanism and judged the replacement in aggregate:

- piece 1 (shrinkage) blended every cell toward a population mean --
  exactly the opposite of a discrete, provable local override, and it
  erased the hand-built policy's per-class rule everywhere, not just where
  a better rule was proven;
- pieces 2-4 (ranking, class-balanced ranking, joint scorer) replaced the
  hand-built policy's decision surface system-wide;
- piece 5 (active learning) only changed training data order.

None of them asked: *is there a specific, narrow context where a different
action provably beats the hand-built default, with a margin tight enough
to trust discretely?* That is the one idea from the shared plan (its
section 5.2-5.3: causal principle utility, lower-confidence-bound
promotion, protected-subgroup regression guard) that is both genuinely new
relative to Phase VII and cheap to test rigorously with data already on
disk -- no new model architecture, no deep-learning infrastructure this
project does not have.

**Explicitly out of scope**, and why: the shared plan's multimodal 3D
backbone redesign, a synthetic-corruption-trained independent verifier
network, specialist-to-generalist curriculum models, and parameter-efficient
adapter infrastructure are each a multi-month research programme on their
own, requiring training infrastructure (new 3D networks, controlled
corruption generation, multi-domain curricula) this project's compute
budget and timeline do not support. Building a half-working version of any
of them would not meet this project's own bar for a result worth reporting.

## The mechanism

For each (failure class, size-bin) cell [tertile bins on `log_volume`,
edges fit on training folds only, same convention as
`phase7_shrinkage.py`]:

1. For every gated component, find its **hand-built pick**: the first
   strategy in `phase4_policy_noban.json`'s ranked list for that component's
   class that was actually measured for that component (exactly replicating
   `phase5_agent.py`'s own hand-built scoring, `len(order) - order.index(strat)`).
2. For every other strategy measured on the same components, compute the
   **paired** delta against the hand-built pick -- same component, both
   actions measured -- restricted to the cell.
3. Bootstrap (2000 resamples) the paired delta's mean and its 2.5th-percentile
   lower confidence bound (LCB), and the **win rate** (fraction of paired
   components where the alternative beats the hand-built pick).
4. **Promote** the cell's best alternative action only if: LCB > 0 (the
   causal effect excludes zero on the positive side), win rate >= 60% (a
   protected-regression guard -- a positive mean driven by a few large wins
   and many small losses is not promoted), and at least 20 paired components
   support the estimate.
5. Everywhere else, the hybrid policy is **byte-identical** to the hand-built
   policy's own scoring -- no blending, no smoothing.

Selection and promotion happen on the four training folds of each outer
fold (3,4,5,6,7) only; the held-out fold never influences which cells are
promoted. The final production artefact promotes cells using all
out-of-fold data, exactly as every Phase VII piece's final artefact was fit.

## Falsification

- **Per-cell**: a promoted override is falsified if its cross-fitted
  held-out paired delta (the outer fold that did not select it) does not
  agree in sign with the LCB that promoted it.
- **Phase-level**: falsified if the hybrid policy's real, case-level,
  cross-fitted agent result (same `run_audit_oof.sh` recipe as every other
  Phase VII/this-phase number) does not beat the hand-built policy's
  +0.0065 [+0.0025, +0.0118], or if it breaches the -0.01 lesion-sensitivity
  safety budget.

## Discipline carried over

Same as Phase VII: cross-fitted on Phase IIIb folds, validation a check
only, test untouched unless a configuration survives OOF and validation, no
existing report/figure/script modified, new files only.
