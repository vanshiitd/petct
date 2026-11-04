# Phase XV: out-of-distribution abstention, the terminal phase -- pre-registered

Written before any Phase XV number is computed. Branch `phase15-ood`, from
`phase14-calibrated`.

## 1. Addressing the forking-paths caution

Phases XIII and XIV were each designed after reading the previous phase's
out-of-fold (and, for XIII, validation) results. A design chosen that way
fits the noise in those splits. **Out-of-fold and validation numbers from
Phase XIII on are therefore not evidence of a win, and this report will not
present them as such.** This plan contains the forking with six commitments.

**(a) A full ledger of what was tried.** Phases VII-XIV each had
pre-registered criteria, and none claimed a win:

| phase | what was learned | configs searched |
|---|---|---|
| VII-X | learned policies, vetoes, class gates | (see their plans) |
| XI | experience-trained gate | 25 |
| XII | KL-anchored additions | 16 |
| XIII | context detector | 1 |
| XIV | calibration, ensemble, guards | 24 |
| XV | XIV + one out-of-distribution rule | 24 (the same grid) |

**(b) Test status, stated exactly.**
- Earlier agents (Phases V-VI, `fe_test_*`, `agent_test_*`, `af_*_test`,
  `p6_*_test`) were scored on test in October.
- **No policy or detector from Phase VII on has ever been run on test**, and
  no hand-built test run (`au_test_attr_policy`) exists.
- Both arms share the hand-built strategies and gate. Any earlier
  test-informed choice in that shared policy is common to both arms, and the
  paired comparison isolates the detector change.

**(c) A minimal change.** Exactly one thing changes from Phase XIV: the
out-of-distribution rule below. The grid, the admissibility rule, the
calibrators, the heads, the features and the endpoints are frozen as
committed in fee8132 and 2e3f241.

**(d) This is the terminal phase.** After Phase XV, the question "does a
learned detector/policy beat the hand-built policy" is **closed** for this
thesis.
- **Test is read exactly once, whatever the out-of-fold result**, provided
  the validation *safety* check passes (section 4). Gating test on
  out-of-fold, as in earlier phases, would invite yet another iteration.
- No later phase may revisit this question on test.

**(e) A test lock.** Before the test agent runs, a commit records:
- SHA-256 of `phase15_ood_test.npz`, `phase11_eag_hb_test.npz` and
  `cand_cv_test_pos.csv`;
- the two exact commands.

The test result is reported whatever it is.

**(f) What would count.** Only criterion 5 on test is the confirmatory
answer. Out-of-fold and validation are reported as descriptive.

## 2. The out-of-distribution rule

A model trained on cases with at most ~140 predicted components has no
evidence about a case with 487. The rule makes the detector **abstain**
there. This is the standard selective-prediction response: hand the case to
the reference.

- **Support.** Compute each training case's number of predicted components
  (`n_pred_case`). The support fence is **Q3 + 3 x IQR** of those counts
  (Tukey's far-out fence). For outer fold k, it is computed on the training
  folds T; for production, on all out-of-fold cases.
- **Deployment.** For a case above the fence, every component uses the
  hand-built detector (old, uncalibrated) at the 0.90 gate. Other cases follow
  the Phase XIV configuration.
- **Training.** The rule is applied consistently. Cases in T above T's fence
  are:
  - left out of isotonic calibrator fitting;
  - scored with the hand-built decision in the admissibility statistics.

  The stage-1 heads are unchanged, so the route checks still hold.

## 3. Everything else

This is unchanged from `reports/phase14_calibrated_detector_plan.md`:
- the 24 configurations;
- admissibility: lesions lost <= hand-built, and tumour-positive
  `d_qdice` >= hand-built, both on T;
- the tie-break and the rollback;
- deployment through `--pfail-table` with `--min-pfail 1e-6`.

## 4. Endpoints

These are unchanged from Phases XIII-XIV, paired against the hand-built run.

There are two superiority routes, so each is tested at a Bonferroni level
of two-sided 0.025.

1. **FP-clearance superiority.** McNemar p < 0.025.
2. **Dice superiority.** dDice > 0, with bootstrap p < 0.025.
3. **Dice non-inferiority.** Lower 95% bound >= -0.0025.
4. **Lesion non-inferiority.** Lower 95% bound >= -5.
5. **Beats the hand-built policy overall.** (1 or 2) and 3 and 4.

**Validation safety check, required for test:**
- dDice >= -0.0025;
- lesion hits retained >= -1;
- no new false positives.

Unlike earlier phases, "cleared >= the hand-built policy's" is dropped: it is
an efficacy check, not a safety check.

**Test.** If the production configuration rolls back entirely, test is the
hand-built policy against itself. That is reported as "no learned change
deployed" and not as a win or a loss.

## 5. Descriptives

- The fence and the abstaining cases, per fold and in production.
- The selection and admissibility table per fold.
- The Phase XIV overconfidence table, recomputed on in-support cases.
- The subgroup table.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
