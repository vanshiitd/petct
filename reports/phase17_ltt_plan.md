# Phase XVII: certified safe improvement by Learn-then-Test risk control -- pre-registered

Written before any Phase XVII number is computed. Branch `phase17-ltt`, from
`phase16-combined`.

## Status and disclosure

This phase is post-terminal, run at the user's request. The work is now
aimed at a paper: "certified safe self-improvement for segmentation
refinement".

**What was already seen.** Every out-of-fold and validation result through
Phase XVI was read before this design and its constants were chosen. In
particular, Phase XVI showed that the best addition set (`new_cal`, 0.95)
lost 4 lesions over 186 additions.

**Consequences:**
- the out-of-fold certification below demonstrates the method; it is not
  confirmation;
- the confirmatory evidence is the AutoPET test split, which has not been
  read for any policy or detector from Phase VII on. It is read **once**,
  under the lock in section 6;
- the forking ledger grows by one phase. Its family is the 88 candidates of
  section 3.

## 1. What changes from Phase XVI

Phase XVI promoted an addition set only if it lost at most 2 lesions in
total. That is an ad hoc count with no stated error guarantee. Phase XVII
replaces it with a **statistical guarantee on the population harm rate**,
using Learn-then-Test (LTT; Angelopoulos, Bates, Candès, Jordan, Lei).

Everything else is Phase XVI's union and is unchanged:
- the hand-built anchor;
- the context detector;
- the isotonic calibration;
- the agreement and solo guards;
- the Q3 + 3 IQR out-of-distribution fence.

## 2. Risk

The unit is the **patient**: there are 630 patients in folds 3-7, and the
folds are grouped by patient. For a candidate update λ, define the patient
harm indicator:

```
H_i(λ) = 1  if any component that λ adds (beyond the hand-built actions),
            in any scan of patient i, is harmful
         0  otherwise
```

A component is harmful under the Phase XI definition (`outcome()`): either
- a true lesion is lost, or
- the patient is tumour-positive and Dice drops by more than 0.01.

Patients with no addition have H_i = 0.

The **risk** is R(λ) = E[H(λ)]: the probability that the update harms a new
patient.

## 3. Test

**Null hypothesis.** For each λ, H_λ: R(λ) > δ. The p-value is the exact
binomial lower tail:

```
p_λ = P( Bin(n, δ) <= k_λ ),   n = 630 patients,   k_λ = Σ_i H_i(λ)
```

**Candidates.** There are 8 chains, one per detector × guard:
- detector ∈ {`new_cal`, `ens`};
- guard ∈ {none, solo, agree, agree+solo}.

Within each chain, the gate takes the values 0.99, 0.98, 0.97, 0.96, 0.95,
0.94, 0.93, 0.92, 0.91, 0.90, 0.85, **tested in that order** (fixed-sequence
testing). Testing stops at the first non-rejection. Each chain is tested at
level α/8 (Bonferroni across chains).

That gives 88 candidates. Family-wise error is at most α.

**Primary constants:**
- δ = 0.01: at most 1 in 100 patients harmed by the update;
- α = 0.05.

These are the "safe level" the user asked for. δ is a clinical tolerance
chosen before Phase XVII computes anything. It was not tuned to make a
candidate pass.

**Promotion.** A candidate is promoted only if all three hold:
1. it is certified (H_λ is rejected);
2. it adds n ≥ 20 components (Phase XII);
3. the patient-clustered bootstrap 95% lower bound of mean `d_qdice` per
   added component is > 0 (Phase XII `stats`).

Conditions 2 and 3 can only reduce promotions, so the guarantee stands.

**Selection.** Among promoted candidates, take the largest summed `d_qdice`.
Ties go to the higher gate, then `ens`, then the stronger guard. If nothing
is promoted, **roll back** to the hand-built policy.

## 4. Guarantee (single round)

> **Theorem.** Assume patients are exchangeable, and the candidate decisions
> for a patient do not depend on that patient's outcomes. Then
> P( R(λ̂) > δ ) <= α, where λ̂ is the deployed update. The rollback has
> R = 0.

Proof: LTT. Each p_λ is super-uniform under H_λ. Fixed-sequence testing
inside a chain plus Bonferroni across chains controls the family-wise
error. λ̂ is either a rejected hypothesis or the rollback.

The assumption holds approximately. The stage-1 heads and calibrators are
cross-fit, so each patient is scored by models not fitted on it. The fence,
one scalar, is computed from all out-of-fold cases. This is disclosed.

Multi-round (lifelong) certification with error budgeting across rounds is
a separate addendum, committed before its numbers.

## 5. Descriptives

All of these are descriptive; no decision depends on them.
- The risk-benefit curve: certified sets and best summed `d_qdice` for
  δ ∈ {0.005, 0.01, 0.02, 0.03, 0.05} and α ∈ {0.05, 0.10}.
- Every candidate's k, p-value and benefit statistics.
- A power note: the smallest k certifiable at n = 630.

## 6. Deployment, endpoints, test

Deployment is as in Phase XVI. Production detectors and calibrators are
fitted on all out-of-fold data, and the agent runs with:

```
--min-pfail 1e-6 --policy phase4_policy_noban.json --T 10
```

Each table is evaluated by `phase13_eval.py` against the hand-built run.
Endpoints 1-5 are unchanged from Phases XIII-XVI.

**Validation safety check, required for test:**
- dDice >= -0.0025;
- lesion hits retained >= -1;
- no new false positives.

**Test.** If production promotes and validation passes, then:
1. a lock commit records the SHA-256 of `phase17_ltt_test.npz`,
   `phase11_eag_hb_test.npz` and `cand_cv_test_pos.csv`, and both commands;
2. the hand-built test run and this policy's test run are each made
   **once**;
3. endpoint 5 on test is the confirmatory result;
4. the observed test patient-harm rate against δ is reported:
   - a patient is harmed if they lose a lesion hit or, when
     tumour-positive, their Dice drops by more than 0.01, against the
     hand-built run;
   - the exact one-sided 95% upper bound is given;
   - this is descriptive (n is small).

If production rolls back, test is not run.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
