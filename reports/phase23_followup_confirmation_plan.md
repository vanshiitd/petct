# Phase XXIII: confirmation of both certificates on never-read follow-up scans (pre-registered)

Written on 15 Nov 2026, before any follow-up scan is preprocessed, predicted
or scored. Branch `paper-cvpr2027`. No model is trained, re-fitted on new
data or tuned. Every decision rule below is fixed now.

## Why

Every test result in the paper comes from C, which was read for Level 1
before Level 2 was planned (limitation 2). The PSMA cohort contains follow-up
studies that no model, analysis or person in this project has read: 218 later
studies of the 377 patients, of which only the earliest was used.

This phase runs the frozen systems on the follow-up studies of the B and C
patients and tests whether the two certificates hold there.

## Cohort F (fixed now)

- Every study with conversion status `ok` in `D:\data\psma_nifti\ledger.jsonl`
  that is not its patient's earliest study, for patients in B or C of
  `reports/phase19_split.json`.
- **143 studies of 111 patients**: 72 studies of 57 B patients, 71 of 54 C
  patients. None is on the same date as the patient's earliest study.
- Follow-ups of A patients (75 studies) are excluded: the fine-tuned model
  was trained on those patients.

**Relation to the guarantee.** These are later studies of patients whose
earliest scan was used to certify (B) or test (C). They are not new
patients and they are not exchangeable with B: they are a later disease
stage, often after treatment. The test is therefore a **temporal-shift
confirmation**. Theorem 1 does not cover it, and a failure would be reported
as a failure of the certificate to transfer over time, not as a statistical
error of the test.

## Frozen systems (nothing re-fitted)

| arm | definition | source |
|---|---|---|
| M0 | FDG nnU-Net, folds 3–7, 8-flip ensemble, p > 0.5 | `cv_predict.py --mode ensemble`, default trainer and results |
| HB | hand-built agent on M0 (Level-0 deployment f0) | `phase4_policy_noban`, stage-1 detector, gate 0.90 |
| P19 | Level-1 certified agent on M0 | config `new_cal`, gate 0.97, guard `agree+solo` (`phase19_psma.json`) |
| FT | fine-tuned ensemble, p > 0.5 (Level-2 deployment at δ = 0.10) | `nnUNetTrainer_PSMAft`, `Dataset507_PSMA_A`, folds 0–4 |

The Level-1 detectors and calibrators are re-created by the Phase XIX code on
the same data with the same seed. **Check:** the re-created HB and P19 action
tables for C must equal the locked `phase19_psma_hb_C.npz` and
`phase19_psma_p19_C.npz` exactly (keys, probabilities, actions), or the phase
stops.

## Pipeline (no step makes a choice)

1. Resample to the FDG grid and export, exactly as `phase18_psma_prep.py`,
   into new directories on D: (`D:\data\psma_fdggrid_fu`,
   `D:\data\nnunet_raw_fu\Dataset505_AutoPET_MAEprep_v2`).
2. TotalSegmentator, same settings (`run_totalsegmentator.py`, fast, seven
   organs) → `D:\data\totalseg_fu`.
3. M0 and FT maps → `D:\data\cv\psma_fu`, `D:\data\cv\ft_fu`.
4. Candidate tables (`build_candidates.py`, `add_position_features.py`,
   `phase13_context_features.py`) on M0 maps → `cand_psmafu_*.csv`.
5. HB and P19 action tables (`phase23_tables.py`).
6. Lock: hashes of the action tables and the maps manifest, committed before
   any agent run or FT scoring.
7. Agent runs HB and P19 (`phase5_agent.py`, as in the Phase XIX lock), FT
   scored with `phase21_safeswitch.score`.

Scripts read the new directories through environment variables
`PETCT_RAW505` and `PETCT_TOTALSEG`; with them unset every script behaves
exactly as before.

**Exclusions.** A study is excluded only if a pipeline step fails for it
(conversion geometry, out-of-memory, missing CT). Exclusions are listed in
the lock, before any outcome. A patient is analysed if at least one of their
studies is.

## Harm (unchanged) and the unit of analysis

Per study, harm of a system against its reference is the paper's definition:
a lesion hit lost, or tumour-positive and Dice down by more than 0.01.

**Unit: the patient.** A patient is harmed if harmed on **any** of their
follow-up studies. This is conservative for patients with several studies.

## Primary confirmations

| | certificate | system vs reference | H0 | rejects iff (n = 111) |
|---|---|---|---|---|
| **H1** | Level 2, δ = 0.10 | FT vs P19 | R > 0.10 | k ≤ 5 harmed patients |
| **H2** | Level 1, δ = 0.05 | P19 vs HB | R > 0.05 | k ≤ 1 harmed patient |

- Exact binomial, α = 0.05 each. They confirm two separate certificates,
  each issued at α = 0.05, so each is tested at α.
- **Secondary:** Holm over H1 and H2 (family-wise 0.05) is also reported.
  The thresholds for n ≠ 111 (after exclusions) are recomputed with the same
  rule.

## Secondary endpoints

Patient-cluster bootstrap (2,000 resamples of patients; per patient, the mean
over their tumour-positive studies), 95% intervals:

1. **Benefit of Level 2:** Dice FT − P19 on tumour-positive studies;
   superiority one-sided.
2. FT − M0 Dice; P19 − HB Dice.
3. FP volume per study, lesion hits, tumour-negative studies cleared of all
   false positives, for every arm.
4. FT vs P19 harm rate with exact 95% upper bound, and the p-value at
   δ = 0.05 (the Level-2 tolerance that was not certified on B).
5. Subsets: follow-ups of B patients and of C patients; per-study harm rate.

## Success and failure (stated now)

- **H1 met** (≤ 5 of 111): the Level-2 certificate held on never-read scans
  under temporal shift.
- **H2 met** (≤ 1 of 111): the Level-1 certificate held likewise.
- Each failure is reported as such, with the harmed patients described.
- If the 95% interval of endpoint 1 lies above zero, the Level-2 gain is
  confirmed on unseen scans.

## Disk and compute

D: has 82 GB free; this phase writes about 17 GB there and nothing to C:.
GPU: TotalSegmentator (≈ 2.2 h), M0 (≈ 2.2 h), FT (≈ 1.7 h).

## Outputs

`scripts/phase23_*.py`; `reports/phase23_lock.md`, `reports/phase23_result.md`;
`D:\data\reports\phase23_*.json`.
