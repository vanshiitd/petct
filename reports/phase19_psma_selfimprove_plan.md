# Phase XIX: certified self-improvement on PSMA -- pre-registered

Written on 2026-11-08, before any PSMA failure diagnosis, strategy-outcome
table, Swin/nnuf0 PSMA map or Phase XIX number exists.

Branch `phase19-psma-selfimprove`, from `phase18-psma`.

## Question

Phase XVIII showed that the frozen agent transfers **safely** to PSMA: 0 of
377 harmed. It also transfers **uselessly**: dDice +0.0000.

Can the agent then **improve itself on the new domain**, using a few PSMA
patients with labels, and come with a certificate that bounds how many new
patients the update harms? Is the benefit real on held-out patients?

The baselines ask a second question: do the obvious alternatives harm more
patients?

## 1. Patients and split (fixed now)

**Patients.** The 377 PSMA patients of Phase XVIII, earliest study each
(`reports/cases_psma.txt`).

**Split.** Patients are randomly assigned, with seed **2027**, stratified by
radionuclide (18F / 68Ga) x tumour status (`case_gt_positive`):

| part | patients | use |
|---|---|---|
| **A**, adapt | 126 | refit detectors, calibrators and fence |
| **B**, certify | 126 | LTT certification and selection only |
| **C**, test | 125 | locked; read once |

Within each stratum, shuffled patients go to A, B, C in turn. The split is
written to `reports/phase19_split.json` and committed **before** any outcome
table is computed.

**Why thirds and δ = 0.05.** These are stated now. With the unchanged
Phase XVII family (8 chains, α/8), k = 0 harmed among n_B patients
certifies δ only when (1-δ)^n_B <= 0.05/8.

| n_B | smallest certifiable δ (k = 0) |
|---|---|
| 113 | 0.044 |
| 126 | 0.040 |
| 152 | 0.033 |

So 3% cannot be certified with any usable split. The earlier conversation
estimate of "about 3%" was wrong.

## 2. Inputs (computed after this plan is committed)

For the nnU-Net ensemble (primary segmenter), on all 377 PSMA cases:

- **Failure diagnosis.** `phase4_diagnose.py --maps D:/data/cv/psma`, giving
  `diag_psma.pkl`.
- **Measured strategy outcomes.** `phase4_strategy_table.py --split test`,
  giving `strategy_table_psma.csv`. This is the same CPU strategy set as FDG.
  The hand-built policy (`phase4_policy_noban.json`) never selects the GPU
  `roi_repredict` strategy, so the FDG `strategy6` table has no PSMA
  counterpart and none is needed.

These cover all of A, B and C. **Phase XIX selection code reads only A and B
rows** and asserts that it does. The C rows are read only after the lock, by
the evaluation scripts.

## 3. The update (the method, unchanged from Phase XVII)

**Anchor.** The hand-built policy (HB): the frozen FDG stage-1 detector
`phase4_diag_model.pkl` gated at 0.90. Its actions are always kept. The
update can only **add** actions (anchored union, `phase16_combined.union`).

**Adapted pieces, all fitted on A only (plus FDG).**

- **Po_cal.** The frozen HB detector's probabilities, isotonically calibrated
  on A's in-support rows (`cal_fit`).
- **Pn_cal.** The **adapted detector**: the Phase XIII new-feature stage-1
  model refitted on FDG out-of-fold rows (labels from `diag_oof.pkl`) plus
  A's rows (labels from `diag_psma.pkl`).
  - Its calibrator is fitted on A predictions cross-fitted over 5 patient
    folds of A (seed 2027). Each fold is scored by a model trained on FDG
    plus the other 4 A folds.
  - The production model is trained on FDG plus all of A, and the calibrator
    is applied to it.
- **Fence.** Q3 + 3·IQR of A's predicted-component counts per case.
  - In support means count <= fence.
  - Out of support, nothing is added.
- **solo_ok.** The case has more than one predicted component, as before.

**Family.** Unchanged: detector {new_cal, ens} x guard {none, solo, agree,
agree+solo} x gate {0.99, 0.98, ..., 0.90, 0.85}. That is 88 candidates in 8
fixed-sequence chains.

**Outcomes.** For each added component, the measured outcome of the
hand-built pick for its argmax class (`labels_for`, from
`strategy_table_psma.csv`). A patient is **harmed** if any added component
loses a lesion, or if the patient is tumour-positive and the component
lowers Dice by more than 0.01. This is identical to Phase XVII.

**Certification on B.** LTT, exact binomial, H0: P(patient harmed) > δ, with
fixed-sequence testing over gates inside each chain and Bonferroni α/8:

- **δ = 0.05, α = 0.05**, n = 126.

**Selection.** Among certified candidates: n_added >= 20 on B, and the
patient-clustered benefit LCB must be > 0. Take the maximum summed benefit,
with the Phase XVII tie-breaks (`choose`).

**No certified candidate.** Then roll back to HB, which is reported as a
failure of the benefit endpoint.

## 4. Arms on C (each run once after the lock)

All arms are run as
`phase5_agent.py --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json`
on a map directory holding only C's cases (hard links to `D:\data\cv\psma`).

| arm | what it is | selection data |
|---|---|---|
| **HB** | hand-built anchor (reference) | none |
| **P19** | certified adapted update (§3) | A fit, B certify |
| **NAIVE** | same family and inputs; the candidate with the largest summed measured benefit on B, adopted if that sum > 0. No harm test, no n or LCB requirement | A fit, B select |
| **PSEUDO** | pseudo-label self-training: the new-feature detector refitted on FDG (true labels) plus A with **pseudo-labels** (class = 1 where the frozen HB detector gives P >= 0.5). Deployed as anchored union, raw probabilities, gate 0.90, guard none, no fence, no certification. Uses no PSMA labels | A fit |
| **UNCAL** | P19 with calibration removed (raw adapted and raw HB probabilities in `union`), certified and selected on B exactly as P19 | A fit, B certify |

The Phase XVIII P17 arm is reported on C descriptively, from the existing
`p18_psma_ltt` run restricted to C.

## 5. Endpoints on C (primary segmenter)

**Primary (both required for "certified self-improvement confirmed").**

- **S19, safety.** Patient-level harm of P19 vs HB on C, defined as in Phase
  XVIII (`phase18_eval.py`): a lost lesion hit, or a tumour-positive dDice
  < -0.01.
  - Exact binomial, H0: R > 0.05, α = 0.05, n = 125.
  - This rejects iff k <= 2.
  - The exact 95% upper bound is reported regardless.
- **B19, benefit.** P19 vs HB, paired per-case dDice, case bootstrap,
  one-sided p < 0.025 (`phase13_eval.py` endpoint 2).

**Key secondary.**

- Paired FP volume (mL/case) vs HB, with 95% CI.
- Lesion-hit non-inferiority (lower bound >= -5).
- Dice NI (>= -0.0025).

**Baseline comparisons (secondary, descriptive with exact CIs).** For each of
NAIVE, PSEUDO and UNCAL on C:

- harmed patients;
- dDice vs HB;
- FP volume vs HB.

The pre-stated expectation, which can fail, is that NAIVE and PSEUDO harm
more patients than P19.

## 6. Generality across segmenters (step 3)

**Segmenters.**

- **Swin**: `nnUNetTrainer_SwinSmall500`, fold 0.
- **nnuf0**: `nnUNetTrainer_500epochs`, fold 0.

Both are the Phase VI models. Each is run on the 377 PSMA cases with
`cv_predict.py --mode ensemble --trainer <T> --folds 0`, with maps written
to `C:\psma_maps\{swin,nnuf0}`. Then come the same candidate, context,
diagnosis and strategy-table scripts.

**Procedure.** The same split, and the same §3 procedure applied to each
segmenter's own A/B rows. FDG training rows stay the nnU-Net ensemble
out-of-fold rows, because no FDG out-of-fold data exist for these models.

**Arms on C.** HB and P19 only.

**Endpoints.** S19 and B19 per segmenter, as a secondary family with Holm
correction over the two segmenters. The headline claim rests on the primary
segmenter only.

## 7. Risk-benefit curve and validity (step 4; run after the C read)

These are run only after the primary endpoints are read.

**Curve.** For each δ in {0.03, 0.05, 0.10, 0.15, 0.20}:

- certify and select on B, as in §3;
- report on C, from the measured tables: summed benefit, added
  components, harmed patients, and the exact 95% upper bound;
- report the same for each segmenter.

The AutoPET curve from Phase XVII is shown alongside.

**Validity.** A and the fitted detectors are fixed. Over 1000 random 50/50
patient splits of B ∪ C (251 patients, seed 2027):

- select on one half, by LTT (δ = 0.05 and δ = 0.10, α = 0.05) and by
  NAIVE;
- measure the realised harmed-patient rate of each choice on the other
  half;
- report the violation rate P(realised > δ).

The claim is that LTT's violation rate is <= α and NAIVE's is not. This is
descriptive, as in Phase XVII-b.

## 8. Lock and order

1. This plan is committed, followed by `phase19_split.json`.
2. Diagnosis and strategy tables are computed (all cases). Swin and nnuf0
   PSMA maps are built, with their tables.
3. `phase19_selfimprove.py` fits on A, certifies on B and writes the C
   tables for every arm and segmenter. It reads no C outcome.
4. **Lock commit.** It records the SHA-256 of every C table and of the C
   candidate rows, the selected configurations, and the commands.
5. Each arm runs once on C. Primary endpoints are evaluated and reported as
   they come out.
6. The curve and validity analyses follow, and are reported whatever they
   show.

## 9. What would count as failure (stated now)

The following outcomes are failures and will be reported as such:

- **Benefit fails.** Nothing certifies on B (rollback), or B19 is not met
  on C.
- **Safety fails.** S19 is not met on C, meaning more than 2 of 125 are
  harmed.
- **Baselines are not riskier.** NAIVE and PSEUDO are no more harmful than
  P19. In that case the certificate costs benefit and buys nothing visible
  on PSMA.

## Deviations

Any departure is recorded here with its date and reason.
