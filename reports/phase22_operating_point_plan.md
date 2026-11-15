# Phase XXII: certifying the operating point of the fine-tuned segmenter (pre-registered)

Written on 15 Nov 2026, after development on split A and before any number
of this phase is computed on B, C or the follow-up studies. Branch
`paper-cvpr2027`. No retraining; no earlier output is modified.

## Why

Phase XX switched from the deployed system (P19) to the fine-tuned segmenter
thresholded at 0.5 (FT). FT harmed 4 of 126 on B, so the switch was not
certified at δ = 0.05. Phase XXI showed that falling back to the deployed
mask where FT is unsure removes none of those harms.

Phase XX tested one operating point of the fine-tuned model. The threshold
is a free parameter that needs no training. A lower threshold detects more
lesions, and harm is mostly lost lesion detections.

## Development record (split A only)

A is the adaptation split: 126 patients, fine-tuned predictions out-of-fold,
never used to certify or test. Reference on A: the hand-built agent on the
zero-shot maps (`phase22_a_reference.py`; the Level-1 update's detectors were
fitted on A, so P19 itself is not label-free there). Everything tried on A:

1. **Diagnosis** (`phase22_dev_diag.py`). FT at 0.5 harms 2 of 126 on A, both
   by lost hits (13→11, 19→18). 39 lesions are lost across 14 patients.
   They lie inside large deployed components (median 114 mL, 15% tumour);
   FT's maximum probability on them has median 0.000.
2. **Per-patient gate** (`phase22_dev_scores.py`): 36 label-free disagreement
   features between the deployed mask and FT. None ranks the two harmed
   patients in the top 3 of 126 consistently; best AUC for "lost a lesion"
   is 0.77 (number of deployed components). **Abandoned.**
3. **Threshold** (`phase22_dev_thresh.py`), FT_t = {p_FT > t}:

   | t | harmed /126 | mean Dice | hits | FP voxels mL |
   |---|---|---|---|---|
   | 0.50 | 2 | 0.4972 | 2773 | 43.7 |
   | 0.40 | 2 | 0.4980 | 2790 | 46.2 |
   | 0.30 | 2 | 0.4984 | 2799 | 49.1 |
   | 0.20 | 1 | 0.4985 | 2817 | 52.9 |
   | 0.10 | 0 | 0.4982 | 2839 | 59.7 |
   | 0.05 | 0 | 0.4980 | 2868 | 67.4 |
   | 0.02 | 0 | 0.4977 | 2912 | 80.4 |

   This is the only idea carried forward. It rests on two harmed patients,
   so A gives a direction, not evidence that it works.

## What is already known about B and C

- FT (t = 0.5) on B: 4 harmed (3 lost hits: 12→6, 3→2, 16→9; one Dice
  drop). On C: 3 harmed, all lost hits.
- Phase XXI's S_τ on B and C, which only adds voxels inside the deployed
  mask.
- FT_t for t < 0.5 has never been computed on B or C.

## Candidates (frozen)

One chain, most sensitive first: t ∈ (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50).
FT_t = {p_FT > t}, the 5-fold ensemble maps in `C:\psma_maps\ft`, no
post-processing. Reference: P19 masks `p21_masks_{B,C}` (already verified
against the locked tables). Scoring: `phase21_safeswitch.score`. FT_0.5 must
reproduce the locked FT rows.

Harm: unchanged. Against P19, a lesion hit lost, or tumour-positive and Dice
down by more than 0.01.

## Selection on B (n = 126)

Fixed-sequence testing from t = 0.02 upward, exact binomial, H0: R > δ, each
at level α_B, stopping at the first non-rejection. Among the thresholds that
pass, t* is the one with the largest mean Dice on B's tumour-positive
patients. If none passes, rollback (P19). Run at δ = 0.05 (primary) and
0.10, 0.15, 0.20.

**Error budget on B.** B was already used once at Level 2 (Phase XX, α =
0.05, nothing rejected at δ = 0.05). This is a second look, designed knowing
the first look's result. Two levels are reported:
- α_B = 0.05: passes iff k ≤ 2 of 126. Over both looks the error is bounded
  by 0.10, not 0.05.
- α_B = 0.025: passes iff k ≤ 1. Not a formal repair either, since the
  second look was not planned.

**The B result is therefore called a selection, not a certificate.** The
claim rests on the tests below.

## Test on C (n = 125), once, after a lock

C has been read for FT at t = 0.5, never for another threshold. After the B
table and t* are committed with hashes:

- **T1 (primary).** Harmed(FT_t* vs P19) on C, exact binomial, H0: R > 0.05,
  α = 0.05. Rejects iff k ≤ 2 of 125.
- **T2.** Paired Dice difference FT_t* − FT_0.5 on tumour-positive patients,
  bootstrap 95% interval; non-inferiority margin −0.01 (the lower bound must
  exceed −0.01).
- **T3.** FT_t* − P19 Dice (benefit), bootstrap.
- Descriptive: every t on C (harmed with exact upper bound, Dice, FP volume,
  hits).

**Success** = t* exists on B at α_B = 0.05, and T1 and T2 hold.
**Failure** = no t passes on B, or T1 fails (3 or more harmed on C), or T2
fails. Reported as such; the rule is not changed afterwards.

## What may be claimed

- If it succeeds: "a lower operating point, chosen on A, selected on B and
  confirmed on C, meets the 5% tolerance". With the caveats that B and C were
  read before for the 0.5 threshold and that the family was designed after
  Phase XX failed.
- A certificate in the strict sense needs patients read for the first time.
  The 143 follow-up studies of B and C patients can provide that under a
  separate plan (Phase XXIII), using the t* frozen here.

## Outputs

`scripts/phase22_threshold.py` → `phase22_threshold_B.json`,
`phase22_threshold_C.json`; `reports/phase22_lock.md`,
`reports/phase22_result.md`.
