# H2 (removal) and H4 (difficult cases) reanalysis -- pre-registered

Written before any new number in this document is computed. Branch
`h2-h4-reanalysis`, from `main` (after Phase VII, PR #10, unmerged).

## Why this exists

Two hypotheses sit short of a clean positive result, for two different
reasons, and both admit a legitimate re-test without touching the agent or
retraining anything expensive:

- **H2, removal task** (`docs/results/phase3b_cv.md` section 3): "reasoning
  features add over uncertainty" is **not supported** for deciding whether a
  predicted component is a false positive -- the reported contrast
  (`d all` − `b unc`, **logistic** model) is −0.0106 [−0.0289, +0.0103] on
  test, straddling zero. But the one GBT number in that report
  (`d all (GBT)`: 0.8362 CV / 0.8360 test) is far above uncertainty-alone
  logistic (0.7645 CV / 0.7777 test) -- and `scripts/phase4_diag_model.py`,
  the detector actually deployed throughout every later phase, is itself a
  `HistGradientBoostingClassifier`, not a logistic model. `scripts/exp3_detect.py`
  (which produced the H2 numbers) only ever fits **one** GBT, on the full
  feature set -- it never fits a GBT restricted to `b unc` alone, so the
  comparison that would actually test H2 against the deployed model's own
  functional form has never been run.
- **H4, difficult cases** (`docs/results/final_eval.md` section 2): narrowly
  supported. The gradient is monotone and validation agrees in sign, but the
  hardest tertile's confidence interval excludes zero by almost nothing
  (+0.0003 of 0.0512 by Q, n=29; +0.0004 of 0.0451 by `u_stab`, n=48).
  Binning a continuous difficulty score into three tertiles and testing only
  the extreme bin throws away most of the sample's power -- a continuous
  test uses all 353 out-of-fold tumour-positive cases instead of 29 or 48.

Both are **re-analyses of existing per-case/per-component data already on
disk** -- no new agent runs, no new candidate generation, no change to any
deployed model. Both are corrections to how the hypothesis was tested, not a
search for a more flattering number: one asks the fair question with the
architecture that is actually deployed, the other recovers statistical power
a binned test discards.

## H2: the fair GBT ablation

For the removal task (`kind="pred"`), fit a `HistGradientBoostingClassifier`
per feature group (`a conf`, `b unc`, `c reason`, `d all` -- anatomical
set, `z_rel` dropped, `dz_*_mm`/`edge_*_mm` added, exactly as
`phase4_diag_model.py` defines them), hyperparameters chosen by grouped CV
on the out-of-fold fitting split alone (`scripts/exp3_detect.py`'s own
`tune_gbt`, reused unchanged), refit on all of the fitting split, scored on
val and test. Paired bootstrap on test: `d all (GBT)` − `b unc (GBT)`,
10,000 resamples, same `paired_bootstrap` function `exp3_detect.py` already
uses for every other H2 contrast.

**Falsified** (no positive update to H2) if this interval still straddles
zero or is negative. **Supported** if it excludes zero on the positive side.
Either way this is the first time the comparison has been run with GBT on
both sides, and the result is reported regardless of which way it comes out.

## H4: continuous difficulty, pooled power

Using the same out-of-fold tumour-positive cases and the same two difficulty
scores (`Q`, `u_stab`) `final_eval.md` section 2 already computed, fit a
single OLS-style regression of `dDice` on the (continuous, rank-normalised)
difficulty score across all 353 cases, bootstrap the slope's 95% CI over
patients (2000 resamples, same convention as every other interval in this
project). Repeat on validation and test as a check, not a fit.

**Falsified** if the slope's interval straddles zero or is negative (gains
are not larger in harder cases, continuously). **Supported, more precisely
than the tertile test** if the slope excludes zero on the positive side with
a tighter relative interval than the tertile result had.

## Discipline carried over

- No existing report, figure, JSON, prediction, or script from a prior phase
  is modified. This analysis writes new files only.
- Test is read only in the manner it was already read for H2/H4 in the
  reports being re-examined (both hypotheses already report test numbers
  descriptively) -- no new touch of test beyond what the original analysis
  already did.
- Hyperparameter selection (H2's GBT grids) never sees val or test.
