# Phase XX: certified self-improvement of the segmenter on PSMA (pre-registered)

Written before any PSMA fine-tuning, prediction or outcome of this phase
exists. Branch `phase20-segmenter-selfimprove`, from
`phase19-psma-selfimprove`.

## Why

Phase XIX's certified agent gain on PSMA is +0.0068 Dice. The ceiling is the
segmenter, not the agent:
- the FDG nnU-Net ensemble (M0) reaches Dice 0.10 on PSMA zero-shot;
- it predicts about 730 mL of false-positive volume per scan;
- the agent can only delete or keep predicted components, so it cannot recover
  tumour that M0 misses.

Phase XX applies the same certify-before-deploy protocol one level up: the
segmenter itself is adapted on A, certified against the current deployment on
B, and tested once on C.

## Split

Unchanged: `reports/phase19_split.json`, with A/B/C = 126/126/125 patients.
- **A** trains the segmenter.
- **B** certifies. No training and no tuning use B.
- **C** is read once per arm, after a hash lock.

C has already been read for the Phase XIX arms. No Phase XX choice is made on
C: the recipe below is fixed now, and selection uses B only. The paper must
disclose this reuse of C.

## Fine-tuning recipe (fixed now, not tuned)

| item | value |
|---|---|
| data | Dataset507_PSMA_A: A's 126 patients, MAE-prep inputs and labels, exactly as Phase XVIII exported them (hard links to Dataset505 `imagesTs` / `labelsTs_prep`) |
| plans | `nnUNetPlans_MAE`, 3d_fullres, copied from Dataset505 (same patch size, spacing and network) |
| folds | 5-fold CV over A, using the same patient-to-fold map as Phase XIX's detector cross-fitting (seed 2027, `i % 5` over a permutation of sorted A) |
| initialisation | A-fold i starts from FDG fold 3+i `checkpoint_final.pth`, **all** weights including the segmentation head |
| trainer | `nnUNetTrainer_PSMAft`: nnU-Net defaults except 100 epochs and initial LR 3e-3 (poly schedule) |
| checkpoint | `checkpoint_final.pth`; no early stopping or checkpoint picking |
| inference | `cv_predict.py`: identical flip fusion and maps. A is predicted out-of-fold, by the fold that never saw it. B and C are predicted by the 5-fold ensemble. |

## Arms (each run once on C, after the lock)

| arm | description |
|---|---|
| M0 | FDG ensemble, zero-shot (**first baseline**). Already measured as the base of `p19_psma_hb`. |
| P19 | current deployment: Phase XIX certified agent on M0. Already measured: `p19_psma_p19`. |
| FT | fine-tuned ensemble, raw, thresholded at 0.5 |
| FT+HB | frozen hand-built agent (`phase4_policy_noban`, stage-1 detector at 0.90) on FT maps |
| FT+P20 | Phase XIX machinery re-run on FT maps (`phase19_selfimprove.py --tag ft_psma`): detectors adapted on A out-of-fold FT maps, LTT at δ 0.05 on B, relative to FT+HB |

## Certification of the switch (on B)

- **Harm, unchanged definition.** A patient is harmed if, compared with P19,
  they lose at least one lesion hit, or they are tumour-positive and their
  Dice drops by more than 0.01.
- **Family.** {FT+HB, FT}. Each is tested at α/2 = 0.025 for H0: R > δ, with
  the exact binomial on 126 B patients.
- **Primary δ = 0.05.** Certified iff k ≤ 1 of 126.
- **Deployed segmenter pipeline (DEP).**
  - If any candidate is certified, DEP is the one with the larger mean Dice on B.
  - If none is certified, DEP is P19 (rollback).
- **Dial (secondary).** The same rule is applied at δ ∈ {0.10, 0.15, 0.20}.
  The certified choice at each δ is reported, with its C outcome.

## Endpoints on C (n = 125)

**Primary.** Holm over E1 and E2; E3 at α 0.05.
- **E1, benefit over the first baseline.** Paired dDice(DEP − M0) on
  tumour-positive patients, bootstrap one-sided p.
- **E2, benefit over the current deployment.** Paired dDice(DEP − P19), same
  test.
- **E3, safety of the switch.** Harmed(DEP vs P19) of 125, exact binomial,
  H0: R > 0.05, α 0.05. It rejects iff k ≤ 2.

If DEP = P19 (rollback), E1 and E2 are evaluated for P19 itself, and E3 is
trivially met with 0 harmed. The rollback is then reported as a failure of
Phase XX.

**Secondary.**
1. FP volume, FN volume, lesion hits, HD95 for every arm. Tumour-free cases
   are cleared of all false positives.
2. FT+P20 vs FT+HB: Phase XIX endpoints S19 and B19 on the FT pipeline.
3. Dial: certified choice per δ, and its realised harm on C with the exact UB.
4. Subgroups: tracer (18F, 68Ga) and vendor, Dice of every arm, harm of DEP.
5. **Uncertified upper reference.** FT and FT+HB vs M0 on C, whatever B
   decided. This is labelled **descriptive** and never as a certified result.

## Failure criteria (stated now)

- E3 fails, i.e. more than 2 of 125 harmed: the switch certificate did not
  hold on C. This is reported as such.
- Rollback at δ 0.05 means no certified switch exists at the primary level.
  The dial and the descriptive FT numbers are then reported, labelled as
  uncertified.
- If FT does not beat M0 on Dice in the descriptive comparison, fine-tuning
  failed.

## Disk and compute

- Preprocessed A: about 7 GB, on C:.
- Training: 5 × 100 epochs, about 9 h on one RTX 5000 Ada.
- Results go to `C:\nnunet_results`, because D: has 7 GB free.
- Maps for A, B and C: about 5 GB, in `C:\psma_maps\ft`.
