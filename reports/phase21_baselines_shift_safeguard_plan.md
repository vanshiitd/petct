# Phase XXI: risk-control baselines, shift stress test and a safeguarded switch (pre-registered)

Written on 14 Nov 2026, before any number of this phase is computed. Branch
`paper-cvpr2027`. Nothing in an earlier phase is modified; every output is a
new file named `phase21_*`.

## What is already known (and so cannot be called blind)

- B and C outcomes have been read for every Phase XIX and XX arm.
- The validity study of Phase XIX (1,000 random halves of B ∪ C) has been run
  for two rules, CSI and naive, at δ 0.05 and 0.10.
- The four B patients harmed by FT against P19 are known. Three lose lesion
  hits; one keeps its hits and loses Dice (0.495 to 0.058).
- On C, FT harms 3 of 125 against P19.

Therefore **every part of this phase is post hoc and retrospective**. Parts A
and B are resampling studies on tables that already exist. Part C designs a
mechanism after seeing the failures it targets. None of the three yields a
certificate. The paper must label them as post hoc, as it does for
Secs. 5.3 and 5.4.

A blind confirmation of Part C is possible only on scans that have never been
read: the PSMA follow-up studies. That would be a separate phase (XXII) with
its own plan and lock, and it is not part of this document.

## Part A: selection rules compared in the validity study

**Data.** The Phase XIX harm matrix `H` and benefit matrix `Bn` (88
candidates × 251 patients of B ∪ C), rebuilt exactly as
`phase19_eval.py` builds them. The same 1,000 permutations (seed 2027);
the first 125 patients of each select, the other 126 evaluate.

**Reproduction check (must pass before anything else is reported).** The CSI
and naive rows at δ 0.05 and 0.10 must equal those in
`phase19_eval_psma.json` to 1e-12.

**Rules.** Each picks one candidate from the selection half, or rolls back
(risk 0, benefit 0). Where several candidates qualify, the rule takes the one
with the largest summed benefit on the selection half (first index on ties),
as CSI does. With n = 125, k = harmed patients on the selection half,
p = P(Bin(n, δ) ≤ k), α = 0.05:

| rule | a candidate qualifies if | multiplicity control |
|---|---|---|
| NAIVE | always | none |
| PLUGIN | k / n ≤ δ | none |
| ZERO | k = 0 | none |
| UNCORR | p ≤ α | none |
| CRC | per chain, the longest prefix of gates (0.99 downward) whose running-max k satisfies (k + 1) / (n + 1) ≤ δ; then the best over the 8 chains | none across chains; per chain it bounds the *expected* risk only |
| BONF-88 | p ≤ α / 88 | Bonferroni over all candidates |
| CSI | fixed sequence within each chain at α / 8 (`ltt_fast`, unchanged) | fixed sequence + Bonferroni over 8 chains |

CRC is conformal risk control. It assumes a loss that is monotone in one
parameter and controls the expectation of the risk, not the probability that
the risk exceeds δ. The script reports how often a patient's harm indicator
is non-monotone along a chain. CRC is also reported per chain (mean over the
8 chains), where its own guarantee applies.

**δ grid.** 0.03, 0.05, 0.10, 0.15, 0.20. Primary: 0.05 and 0.10.

**Metrics per rule and δ.**
- violation rate: share of splits whose evaluation-half harm rate exceeds δ,
  with a 95% Clopper–Pearson interval for Monte-Carlo error;
- mean and 95th percentile of the evaluation-half harm rate;
- mean summed benefit on the evaluation half;
- rollback frequency and mean number of qualifying candidates.

Resampling is by patient; each patient has one scan, so no clustering is
needed.

**Expectations, stated now.**
- A1. CSI's violation rate is at most α at both primary δ (already known).
- A2. NAIVE, PLUGIN and UNCORR exceed α at δ = 0.05.
- A3. BONF-88 gives no more mean benefit than CSI at either primary δ.
- A4. No prediction for ZERO or CRC.

**What counts against the paper.** If any rule has a violation rate at most α
and a larger mean benefit than CSI at a primary δ, the paper reports that
this rule dominates CSI in this study.

## Part B: certification under a known shift

**Question.** The guarantee assumes certification and deployment patients are
exchangeable. How badly does it fail when they are not?

**Groups** (B ∪ C, metadata only): tracer 18F 142 / 68Ga 109; vendor 1 148 /
vendor 2 103. Four directions: 18F→68Ga, 68Ga→18F, vendor 1→2, vendor 2→1.

**Procedure.** For each direction, 1,000 draws (seed 2028). Each draw splits
the source group at random into halves:
- selection: one half of the source;
- **control**: the other half of the source (exchangeable);
- **shifted**: every patient of the target group.

All Part A rules run on the selection half. Harm rate and violation are
measured on the control and on the shifted set.

**δ.** Primary 0.10. With α / 8, certifying needs n ≥ 49 at δ = 0.10 and
n ≥ 99 at δ = 0.05. The selection halves have 51 to 74 patients, so at
δ = 0.05 CSI must roll back in every draw. δ 0.05, 0.15 and 0.20 are reported
as secondary.

**Metrics.** Per direction, rule and δ: violation rate (control, shifted),
mean harm rate (control, shifted), rollback frequency. Shifted-set violation
is a property of one fixed target group, so it measures the candidates'
actual harm there, not a sampling probability.

**Expectations.** None directional. The result is reported whichever way it
falls. If CSI stays within δ under shift, the paper says the guarantee was
not needed to transfer and that this is an observation, not a guarantee.

## Part C: a safeguarded switch for the segmenter (Level 2)

**Motivation.** Phase XX replaced the deployed output P19 wholesale with the
fine-tuned segmenter FT. That harmed 4 of 126 on B, so δ = 0.05 was not
certified. A switch that keeps some of the deployed output may lose fewer
lesions.

**Mechanism (fixed now).** For a threshold τ, with p_FT the fine-tuned
ensemble probability and P19 the deployed mask:

    S_τ = {p_FT ≥ 0.5} ∪ (P19 ∩ {p_FT ≥ τ})

The new model needs less evidence where the deployed system already fired.
S_0.5 is FT; S_0 is FT ∪ P19. The masks are nested: S_τ grows as τ falls. A
lesion hit by P19 can be lost only if FT gives every P19 voxel on it a
probability below τ; S_0 can lose none.

**Family: one chain, safest first.**
τ ∈ (0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50).

**Inputs, all existing.**
- FT maps: `C:\psma_maps\ft` (B and C by the 5-fold ensemble).
- P19 masks: `phase5_agent.py --save-masks`, re-run with the locked
  tables (`phase20_psma_p19_B.npz` on B, `phase19_psma_p19_C.npz` on C) under
  new out-prefixes `p21B_psma_p19` and `p21C_psma_p19`. The new `_final.csv`
  must equal the locked `p20B_psma_p19_final.csv` / `p19_psma_p19_final.csv`
  in dice, n_hit and fp_ml for every case, or Part C stops.
- Ground truth and scoring: `CaseContext`, the same Dice and lesion-hit code
  as every earlier phase. As a second check, S_0.5 must reproduce the locked
  FT rows (`p20B_ft_hb_base.csv`, `p20_ft_p20_base.csv`).

**Harm.** Unchanged: against P19, a lesion hit lost, or tumour-positive and
Dice down by more than 0.01.

**Selection on B (n = 126).** Fixed sequence from τ = 0 upward, each tested
with the exact binomial for H0: R > δ at α = 0.05, stopping at the first
non-rejection. At δ = 0.05 a τ passes iff k ≤ 2. Among the τ that pass, the
choice τ* is the one with the largest mean Dice on B's tumour-positive
patients. If none passes, or τ* does not beat P19's mean Dice on B, the
result is rollback. The same rule is run at δ 0.10, 0.15, 0.20.

**This is selection, not certification.** B was already used to test FT, and
the mechanism was designed knowing how FT failed on B. The B p-values are
reported for transparency and are not a certificate.

**Order of work.** All of B first: the checks, the τ table, τ*. τ* and the B
table are committed in a lock file with hashes. Only then is S_τ computed on
C.

**Check on C (n = 125), post hoc.** For S_τ* at each δ, and for every τ as a
descriptive curve: harmed of 125 against P19 with the exact 95% upper bound,
mean Dice on tumour-positive patients, FP volume, lesion hits, and the paired
Dice difference to FT and to P19 (bootstrap, 2,000 resamples).

**Expectations.** None. Possible outcomes, all reported:
- S_τ* harms fewer than FT on C at little cost in Dice: the safeguard is
  promising and needs blind confirmation;
- S_τ* keeps so much of P19 that Dice falls well below FT: the safeguard
  trades too much benefit for safety;
- nothing passes on B at δ = 0.05: rollback, as in Phase XX.

**What the paper may claim.** Only "a post hoc safeguard, selected on B,
checked on C". Not a certificate at 5%, whatever the C count is.

## Outputs

- `scripts/phase21_rules.py` → `D:\data\reports\phase21_rules.json` (Parts A, B)
- `scripts/phase21_safeswitch.py` → `phase21_safeswitch_B.json`,
  `phase21_safeswitch_C.json` (Part C)
- `reports/phase21_lock.md`, `reports/phase21_result.md`

## Compute

CPU only for A and B (minutes). Part C: two agent runs of about 15 minutes
each to save the P19 masks (about 1 MB per case), then mask arithmetic.
