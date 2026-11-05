# Phase XVII-b: lifelong certification and the validity check -- pre-registered addendum

Written after the Phase XVII certification result was seen (commit of
`phase17_ltt.py`) and before any XVII-b number is computed. Both analyses
are **descriptive**: no deployment and no test read depends on them.

## 1. Lifelong (multi-round) certification

**Setting.** In downstream use, the agent improves itself in rounds:
1. retrain on all data seen so far;
2. propose the 88-candidate family of Phase XVII;
3. certify on data the round's models never saw;
4. deploy, or roll back.

Round r is tested at α_r = α · 2^(-r), so Σ_r α_r < α.

> **Theorem (lifelong).** Assume that in every round, the round's candidate
> family is fixed by data independent of that round's certification
> patients, and that those patients are exchangeable with future patients.
> Then
> P( some round deploys an update with R > δ ) <= Σ_r α_r < α.

Proof: within a round, the Phase XVII LTT bound holds at α_r. Apply the
union bound over rounds. The bound needs no independence *between* rounds,
so certification patients may be reused as training data in later rounds.

**Simulation on folds 3-7.** The arrival order is the fold order.

| round | train (detectors, calibrators, fence) | certify on | α_r |
|---|---|---|---|
| 1 | folds 3, 4 | folds 5, 6, 7 | 0.025 |
| 2 | folds 3, 4, 5 | folds 6, 7 | 0.0125 |

Nesting is as in the Phase XVI outer folds:
- stage-1 heads come from inner cross-fits within the training folds;
- the calibrators are fitted on those cross-fits, in support only;
- certification patients are scored by heads fitted on all the training
  folds.

The anchor in each round is the hand-built rule, applied with that round's
old detector. The risk is the Phase XVII patient harm indicator, against
that anchor.

**Reported for δ ∈ {0.02, 0.03, 0.05}:**
- per round: certified count, deployed configuration, n added, summed
  `d_qdice`, and harmed patients;
- whether benefit rises from round 1 to round 2 (the self-improvement
  claim);
- the sample-size bound: patients needed to certify with k = 0,
  n >= ln(α_r / 8) / ln(1 - δ).

δ = 0.01 is not run here. A round certifies on at most 378 patients, which
is below the k = 0 requirement of 517 at α_1/8. That figure is arithmetic,
not data.

## 2. Validity check (empirical coverage of the guarantee)

**Splits.** 1000 random patient-level 50/50 splits of the 630 out-of-fold
patients, with seed 2026. The Phase XVII candidates are held fixed; their
heads and calibrators are cross-fit.

**Per split:**
- LTT, exactly as Phase XVII, on the calibration half (n = 315), at
  α = 0.05;
- among certified candidates, pick the largest summed `d_qdice` on the
  calibration half. The bootstrap benefit check is omitted for cost; this is
  disclosed;
- record the chosen candidate's harm rate on the evaluation half.

**Comparator:** a naive selector with the largest summed `d_qdice` on the
calibration half and no safety test.

**Reported for δ ∈ {0.02, 0.03, 0.05}** (δ = 0.01 needs n >= 517 for k = 0):
- violation frequency, P(evaluation harm rate > δ), certified vs naive;
- mean evaluation harm rate and mean evaluation benefit;
- the rollback frequency.

The evaluation half is itself finite (n = 315). Its rate therefore estimates
R with noise, and violation frequency is a conservative, noisy proxy for
P(R > δ). This is stated with the result.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
