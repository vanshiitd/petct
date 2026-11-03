# Phase XI: the experience-trained action gate -- pre-registered

Written before any Phase XI model is fitted, any agent is run, or any
validation or test number for this configuration is computed. Branch
`phase11-trust`, from `phase9-class-gates`.

## Why this, and what is already known

Phases VII-X are the evidence this design is built on, and they are disclosed
here because they were all read on out-of-fold data:

- **Strategy choice has no headroom.** Every learned selector's whole deficit
  against the hand-built policy is two or three patients (synthesis, section 1).
- **The decision that matters is whether to act, not how.** Below the 0.90
  gate the hand-built pick still pays on average, but 24-37% of those actions
  delete a real lesion (Phase IX). Inside the gate, 6.4% do (Phase IX).
- **That decision is currently made by a model that never saw an action's
  outcome.** `p_failure` is trained on ground-truth diagnosis labels, one
  component at a time, with no case context. Harmful removals are interior,
  small, and in cases the detector already knows contain tumour (Phase X).
  `edge_body_mm` alone separates harmful from beneficial removals at AUROC 0.84.

So the self-improvement step this phase tests is: **the agent learns when to
act from the measured outcomes of its own past actions**, with case context
the per-component detector never had. The strategy choice stays the
hand-built policy's, which nothing has beaten.

Because the reasons above were found on out-of-fold data, **out-of-fold
results here are a development check, not confirmation.** Validation and test
are confirmatory.

## The model

**Unit.** Every predicted component with a cross-fitted class and `p_failure`
(`audit_crossfit.npz`, 6462 rows). Its candidate action is the hand-built
policy's pick for its argmax class (`phase4_policy_noban.json`), exactly as
`phase9_class_gates.load_components` defines it.

**Label (experience).** The measured outcome of that pick in
`strategy_table_oof.csv` / `strategy6_oof.csv`. `harm` is defined as in Phase IX:
a lesion lost, or a tumour-positive case whose Dice drops by more than 0.01.

**Features (all observable, no ground truth).**
- the 26 observable features of the Phase IV detector;
- the 8 stage-1 class probabilities;
- six case-context features, computed from the stage-1 probabilities of the
  *other* components in the same case:
  - `ctx_min_pmax`: the lowest max-class probability among them, so a low
    value means the detector is confident the case has a real lesion;
  - `ctx_n_conf`: how many of them have max-class probability < 0.3;
  - `ctx_n_comp`: the number of predicted components in the case;
  - `ctx_vol_rank`: this component's volume rank within the case, in [0, 1];
  - `ctx_suv_rel`: `suv_max` relative to the highest `suv_max` among the
    confident (max-class probability < 0.5) others, NaN if there are none;
  - `ctx_dz_conf`: the smallest |`centroid_z` difference| to a confident
    other, NaN if there are none.

**Harm model.** `HistGradientBoostingClassifier` with the hyperparameters every
detector in this project uses (`max_depth=3`, `learning_rate=0.08`,
`max_iter=200`, `random_state=0`). It is fitted on components with
max-class probability >= 0.5 and a measured pick, which is the region where
any gate below could act. No tuning.

**Decision.** Act on a component if and only if its max-class probability is
>= `g` **and** its predicted harm is <= `h`. The grid is
`g` in {0.90, 0.80, 0.70, 0.60, 0.50} and `h` in {0.02, 0.05, 0.10, 0.20, 1.00}.
The point (`g` = 0.90, `h` = 1.00) is the hand-built policy exactly.
Recovery candidates keep the unchanged 0.90 gate.

## Honest nesting

For each outer fold `k` in (3, 4, 5, 6, 7), with training folds `T` = the other
four:

1. **Stage 1, nested.** Class probabilities for rows in `T` come from 4-fold
   CV within `T`. Rows in `k` are scored by stage-1 models fitted on all of
   `T`. Fold `k`'s labels never enter anything that scores fold `k`.
2. **Harm model.** Predicted harm for rows in `T` comes from 4-fold CV within
   `T`, on the step-1 features. This is the standard stacking compromise: an
   inner row's stage-1 feature saw the other inner folds' labels. The
   compromise is disclosed, and it cannot reach fold `k`. Rows in `k` are
   scored by a harm model fitted on all of `T`.
3. **Selection on `T` only.** Choose the (`g`, `h`) that **maximises the total
   measured `d_qdice` of the acted set**, subject to:
   - **(a)** lesions lost in the acted set <= 0.5 x the hand-built policy's
     lesions lost on the same rows (gate 0.90, no harm filter);
   - **(b)** summed `d_qdice` on tumour-free cases >= the hand-built policy's,
     so false-positive clearance is never given up.

   Ties go to the larger `g`, then the smaller `h`. If nothing passes, use the
   hand-built point.
4. Apply the chosen (`g`, `h`) to fold `k`'s scores. The decisions over the
   five outer folds form the honest out-of-fold table.

**Production model (validation, test).** Stage 1 is `phase4_diag_model.pkl`,
fitted on all 738 out-of-fold scans, which is what the agent already uses on
validation and test. The harm model is fitted on all out-of-fold rows, using
`audit_crossfit.npz` as the stage-1 features. (`g`, `h`) is selected the same
way, on all out-of-fold rows, with 5-fold cross-fitted predicted harm.

## Deployment

**No change to `phase5_agent.py`.** The decisions are written into a new
`--pfail-table` npz:

- for an acted component, the row keeps its stage-1 probabilities, so its
  class and hand-built pick are unchanged;
- for any other component, the row is zeroed;
- recovery probabilities below 0.90 are zeroed too.

The agent then runs with `--min-pfail 1e-6`, so this table is the only gate.
Every other flag matches the hand-built reference run (`au_oof_attr_policy`):
`--policy phase4_policy_noban.json --T 10`, no Q gate, no ROI.

**Smoke equivalence check.** Before the real runs, a table built at the
hand-built point (`g` = 0.90, `h` = 1.00) must reproduce `au_oof_attr_policy`
on 10 cases. If it does not, the deployment route is wrong, and that is fixed
before anything else.

## Endpoints

All are paired, case by case, against the hand-built run (`au_{split}_attr_policy`).

1. **Lesion-safety superiority.** Lesion hits retained (EAG - hand-built),
   summed over tumour-positive cases, with a case bootstrap: the lower 95%
   bound must be > 0.
2. **Dice non-inferiority.** Paired dDice (EAG - hand-built): the lower 95%
   bound must be >= -0.0025.
3. **False-positive clearance non-inferiority.** Tumour-free patients
   cleared must be at most 2 fewer than the hand-built policy's.
4. **Win (strong).** 1-3 hold **and** paired dDice point estimate >= 0. In
   words: safer than the hand-built policy at no Dice cost.
5. **Secondary: against the final agent** (`au_{split}_finalagent`, Q gate).
   Report whether the EAG agent has higher dDice vs M0 and no more lesions
   lost.

**Validation** (38 tumour-positive cases) is a direction check:
- lesions retained >= 0;
- paired dDice point estimate >= -0.0025.

**Test** is read **once**, and only if criteria 1-3 hold out-of-fold and
validation passes its direction check. It is read for this configuration and
for one new hand-built test run, since none exists yet. If any of these fails
first, test stays untouched.

## Trustworthiness reporting (descriptive, pre-specified)

- **Calibration.** For every stage-1 class and for the harm model, both
  cross-fitted:
  - expected calibration error, with 10 equal-mass bins;
  - Brier score;
  - a reliability table.
- **Certified harm.** In the acted set (held-out decisions), the observed
  harm rate with a one-sided 95% Clopper-Pearson upper bound, against the
  hand-built policy's.
- **Subgroup no-harm.** Paired dDice vs M0 and vs hand-built, and lesions
  lost, by:
  - tumour-free versus tumour-positive;
  - tertile of ground-truth lesion count (boundaries on out-of-fold);
  - tertile of M0 Dice (boundaries on out-of-fold).

  The flag raised is any subgroup whose dDice vs M0 has an upper 95% bound
  < 0. Clinical metadata (sex, age, diagnosis) is not available locally, so
  these are proxy groups and are reported as such.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
