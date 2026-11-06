# Phase XVIII: external confirmation on PSMA PET/CT -- pre-registered

Written before any PSMA prediction, candidate table or agent run exists.
Branch `phase18-psma`, from `phase17-ltt`.

## Question

The Phase XVII certified agent was developed on FDG. Is it **safe, and no
worse than the hand-built policy**, on a new tracer, new scanners and new
patients it has never seen?

The agent is frozen. That means the deployed configuration (`new_cal`,
gate 0.98, guard none), its detectors and calibrators fitted on all FDG
out-of-fold data, and the out-of-distribution fence of 34 components per
case. **Nothing is refitted or re-selected on PSMA.**

## Data

**Source.** PSMA-PET-CT-Lesions v2 (`reports/psma_dataset.md`): 595
converted studies from 377 patients.

**Unit.** One study per patient: the **earliest** study, with ties broken by
UID suffix. That gives 377 patients, so patients are independent. It also
fits the disk budget, since all 595 resampled studies do not fit. This is
fixed now, before any data is looked at.

**Grid.** Each study is resampled onto the FDG grid: 2.0364 x 2.0364 x 3.0
mm, keeping the PSMA scan's origin and direction.
- PET (SUV) and CT: linear interpolation (CT filled with -1024);
- mask: nearest neighbour.

Ground truth is the resampled mask, used identically for every arm. The
models only accept this grid: the MAE preparation refuses non-unit zoom.

**Pipeline.** Unchanged FDG code throughout:
1. `export_nnunet_maeprep.export_case` writes into Dataset505 `imagesTs`,
   `labelsTs_prep` and `inverse` (new files only, `PSMA_*` names);
2. `cv_predict.py --mode ensemble` runs the five FDG folds, giving maps in
   `D:\data\cv\psma`;
3. `build_candidates.py`, `add_position_features.py` and
   `phase13_context_features.py` build the candidate tables.

**Base model.** The FDG-trained nnU-Net ensemble with no PSMA training.
This is a zero-shot domain shift, and it is deliberate: the question is
about the refinement agent, not the segmenter.

## Arms (each run once, after the lock)

| arm | table |
|---|---|
| hand-built (HB) | stage-1 detector (`phase4_diag_model.pkl`) gated at 0.90, built as `phase11_eag_hb_{split}` |
| Phase XVII (P17) | the frozen production update over HB, built as `phase17_ltt.py` production |

Both run with:

```
phase5_agent.py --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json
```

`--split test` is used because PSMA, like test, is scored by the full
(non-fold) models.

## Endpoints

All endpoints compare P17 with HB, paired per patient.

**Primary (confirmatory).**

- **S, safety.** A patient is *harmed* if, against HB:
  - they lose at least one lesion hit, or
  - they are tumour-positive and their Dice drops by more than 0.01.

  Test H0: R > δ = 0.01 with the exact one-sided binomial, at α = 0.05,
  n = 377.

  **Power, stated now:** this rejects only if k = 0, since 0.99^377 =
  0.023, and k = 1 gives p = 0.11.

  The exact 95% upper bound on R is reported whatever the outcome.
- **D, Dice non-inferiority.** Paired dDice lower 95% bound >= -0.0025
  (case bootstrap, as in `phase13_eval.py`).
- **L, lesion non-inferiority.** Lesion hits retained, lower 95% bound
  >= -5.

**Confirmed safe transfer** = S and D and L.

**Secondary.**
- The Phase XIII-XVI endpoints 1, 2 and 5, using `phase13_eval.py`
  criteria. With an expected ~40 tumour-free patients, FP-clearance
  superiority has little power.

**Descriptive** (no decision depends on these):
- base-model performance on PSMA (Dice, FP and FN volume, lesion hits);
- the number of out-of-fence (abstaining) cases;
- components added by P17 over HB;
- per-patient changes;
- subgroups by radionuclide (68Ga / 18F), scanner manufacturer and age
  tertile, from the clinical TSV. This is the fairness view.

## Lock

The candidate tables carry TP/FP status, which reveals base-model
performance but no policy outcome. They may be built and inspected for
integrity before the lock.

The lock commit then records:
- SHA-256 of both PSMA tables and `cand_psma_pos.csv`;
- both commands.

Each agent runs once after it, and every result is reported as it comes
out.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
