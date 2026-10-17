# Phase V analysis plan (pre-registered)

Written and committed **before any Phase V number is computed**. Branch
`phase5-memory`, from `phase4-agent`. Where a criterion below fires, it will be
reported as it fired, including when the verdict is negative.

Phase IV is the immediate input: `reports/phase4_analysis_plan.md` and the report
at `D:\data\reports\phase4_agent.md`.

---

## 1. What Phase V changes, and why

Phase IV established four things that this phase is built to act on.

1. **Q cannot adjudicate single actions** - 37.1% of actions with dQ > 0 improve
   Dice, against a 60% pre-registered threshold. The reflection gate was reported
   unusable.
2. **The diagnostic head's own confidence can.** Gating on `p_failure >= 0.90`
   turned out-of-fold dDice from -0.0045 [-0.0113, -0.0001] into
   +0.0020 [+0.0009, +0.0033] and 105 lost lesions into none. That result was
   **post-hoc**; here it is pre-registered.
3. **The hand-built policy was weak.** It selects one strategy per failure class
   by mean dDice over the measured table, and the pre-registered gaming ban kept
   the two strategies with the smallest samples and banned the seven measured on
   thousands.
4. **`roi_repredict` was never run inside the loop.** It is the only strategy
   whose class profile matches the diagnosis and the only one approaching the Q
   threshold (58.6%), but Phase IV measured it only as a per-component
   counterfactual.

Phase V therefore: replaces the hand-built policy with a **learned action-value
model** (Experiment 5), pre-registers **`p_failure` gating**, puts
**`roi_repredict` in the agent's REGISTRY** and measures it at case level, and
**raises the per-case action budget T**.

## 2. Hypotheses

- **H5.** Experience-based strategy memory improves strategy selection: a model
  fitted on measured action outcomes selects better actions than the Phase IV
  hand-built policy, measured by the co-primary endpoints in section 4.

**H5 is the only work-plan hypothesis Phase V tests.** An earlier revision of
this plan also labelled a second question "H6" - whether `roi_repredict` in the
loop improves Dice on tumour-positive cases. That label was wrong: H6 in the
work plan is backbone independence (SwinUNet), which Phase V does not touch. The
question is a legitimate Phase V objective and is kept as **objective B**, but it
is not a numbered hypothesis and no H6 verdict is reported here.

## 3. Data, splits and discipline

Unchanged from Phases IIIb and IV, and non-negotiable here:

- **Out-of-fold** (738 scans, patient-grouped 5 folds) is the only fitting set.
- **Validation** (76) is a check, never a fitting set.
- **Test** (200 scans, 200 patients; 110 tumour-positive, 90 tumour-free) is read
  **once**, at the end, after every threshold and hyperparameter is frozen.
- Every model is evaluated out-of-fold by the fold model that did not see the
  patient. Any threshold is chosen on out-of-fold, checked on validation, and
  applied unchanged to test.
- M0 throughout is the Phase IIIb 5-fold ensemble.

## 4. Co-primary endpoints

Phase IV's primary endpoint - mean Dice over tumour-positive cases - is blind by
construction to this system's one reliable effect: it scored the clearing of every
false positive from 49 of 385 out-of-fold tumour-free patients as exactly 0.0000.
Phase V therefore carries **two** co-primary endpoints.

**Co-primary 1 - segmentation quality.** Mean Dice over tumour-**positive** test
cases (n=110). Paired case-level bootstrap over patients, 2000 resamples.

**Co-primary 2 - false-positive burden.** The **number of tumour-free test
patients carrying at least one false positive** (n=90), compared paired by
patient with **McNemar's test** (exact, two-sided) on the discordant pairs.
**FP volume in mL over all 200 cases** is its supporting measure, by paired
bootstrap, reported alongside but not itself a co-primary.

### Reference

Both co-primaries are assessed against **M0**, the Phase IIIb 5-fold ensemble.
That is the comparator every Phase IV endpoint used and the only one that answers
"does the agent help at all".

As a **pre-registered secondary** comparison, Phase V is additionally compared
against the **Phase IV guarded configuration** (`agent_*_pf90`, tau_Q = 1.00,
`p_failure >= 0.90`), which answers the different question "does learned
selection beat the hand-built policy". H5 is judged on this second comparison;
objective B on the first.

### Success criterion

Phase V succeeds only if **both** of the following hold.

1. **Non-inferiority on both co-primaries.** For Dice, the lower bound of the
   two-sided 95% interval on dDice must exceed **-0.01**. For the tumour-free FP
   count, the Phase V count must not exceed the reference count (non-inferiority
   margin 0 patients, since more tumour-free patients carrying false positives is
   never acceptable).
2. **Superiority on at least one co-primary**, with the two superiority tests
   corrected for multiplicity by the **Holm step-down procedure** at family-wise
   alpha = 0.05. Concretely: order the two p-values, require the smaller to be
   below 0.025 and, if it is, the larger below 0.05.

Failing either condition, Phase V is reported as not successful. Partial results
are reported in full regardless.

## 5. Safety constraint, which overrides everything

**Lesion-level sensitivity non-inferiority, margin -0.01.** Delta lesion
sensitivity against M0 must be at least **-0.01** on out-of-fold and validation
for a configuration to be eligible at all, and is reported on test.

**If the safety constraint fails on test, Phase V is reported as a failure
whatever the co-primaries show.** No configuration that deletes lesions is
reported as a success because it improved an aggregate.

Test lesion sensitivity is never used to select anything.

## 6. Secondary endpoints

Reported for every configuration, none of them selection criteria:

- FP lesions per patient; FN volume (mean over all cases); HD95
  (tumour-positive).
- Improved / unchanged / degraded split, `|dDice| < 0.01` counting as unchanged.
- Lesions removed and recovered by size bin (<0.5, 0.5-2, 2-10, 10-50, >50 mL).
- Actions per case, accepted fraction, runtime per case.
- Per-action mean dDice by strategy and by failure class.

## 7. Experiment 5: the learned action-value model

**Target.** For each measured action, `d_qdice` - the Dice change with an emptied
tumour-free mask scored as 1.0 - as defined and validated in Phase IV section 3.
`d_dice` (nan on tumour-free cases) is kept for the primary endpoint only.

**Training set.** The 34,999 measured actions already on disk: 33,799 CPU edits
(`strategy_table_oof.csv`) plus 1,200 GPU region re-predictions
(`strategy6_oof.csv`), all from out-of-fold cases only.

**Features.** The 26 observable features of `exp3_detect`, plus the proposed
strategy as a categorical, plus `p_failure` and the predicted failure class. No
feature derived from ground truth, and no feature Q is fitted on.

**Model.** Gradient-boosted trees, hyperparameters chosen by a grouped-by-patient
CV grid search on out-of-fold alone, exactly as Phase IIIb's corrected procedure.
No fitting on validation.

**Use in the loop.** The learned model replaces the hand-built policy's ordering:
at each iteration the agent scores every applicable (component, strategy) pair
and takes the highest predicted `d_qdice` above a threshold `eps_hat`, subject to
the `p_failure` gate.

**Falsification.** If the learned model's out-of-fold precision at predicted
improvement - the fraction of its selected actions that actually improve
`d_qdice` - does not exceed the hand-built policy's, Experiment 5 is reported as
failed and the hand-built policy is retained.

## 8. Pre-registered configuration

- **`p_failure >= 0.90`** as a hard precondition on every proposal.
  Pre-registered on the Phase IV out-of-fold and validation evidence; not
  re-tuned.
- **`roi_repredict` in REGISTRY**, evaluated at case level over the full
  out-of-fold set.
- **T = 10** actions per case, raised from 3. Phase IV showed 3 actions cannot
  reach 8.8 spurious and 12.7 missed items per case. T is fixed here and not
  tuned.
- **No Q gate.** Q is not used to accept or reject actions. It may be reported as
  a descriptive quantity only.

## 9. Ablation ladder

Run as genuinely separate configurations, which Phase IV did not do:

1. M0 alone.
2. M0 + `p_failure` gate + hand-built policy (= the Phase IV guarded
   configuration).
3. + learned action-value model, `roi_repredict` excluded.
4. + `roi_repredict` in REGISTRY.
5. + T raised to 10.

Each rung differs from the one above it in exactly one component. The labels in
the evaluation output will name the component, not the iteration index - the
Phase IV output used `M1`/`M2`/`M3` for iteration states and the report initially
mislabelled them as ablation rungs.

## 10. Falsification

- **H5 fails** if the learned model does not beat the hand-built policy on either
  co-primary under the section 4 criterion.
- **Objective B fails** if the Dice co-primary interval does not exclude zero on
  test.
- **The loop is not deployable** if no configuration satisfies section 4 while
  meeting section 5.
- **Learned selection is not usable** if its out-of-fold precision does not
  exceed the hand-built policy's (section 7).
- **`roi_repredict` does not compose** if rung 4 does not beat rung 3 on either
  co-primary - that is, if per-component gains do not survive composition.

## 11. Deviations

Any departure from this plan is recorded in the report with the reason, the date,
and what it changed. Configurations that cannot be executed are reported as not
executed, never silently dropped.

## 12. Provenance of the endpoint change

Phase V adds a second co-primary that Phase IV did not have. A reader is entitled
to ask whether the endpoint was changed after seeing which one Phase IV succeeded
on. The honest answer, recorded here before any Phase V number exists:

**The change is a post-hoc response to a Phase IV finding.** No document
predating Phase IV in this repository or on this machine states the limitation in
advance; a note was believed to exist (`training_runs.md`, 2026-09-25) but no
such file is present on disk or in git history, and the earliest report of any
kind is dated 2026-09-30. The change is therefore justified on its merits - Dice
over tumour-positive cases cannot see false-positive clearance on tumour-free
patients, which is arithmetic rather than an empirical finding - and a reader
should discount it accordingly. It is pre-registered here, before Phase V data,
which is the most this phase can honestly claim.

If a predating record is located, this section will be amended to cite it
precisely, and the amendment will be a visible commit rather than an edit in
place.
