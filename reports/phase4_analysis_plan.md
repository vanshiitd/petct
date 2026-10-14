# Phase IV analysis plan (pre-registered)

Written and committed **before any Phase IV number is computed on the test split**.
Branch `phase4-agent`, from `phase3b-cv`. Everything in Phase IV is learned and tuned on
the 738 out-of-fold (OOF) training scans with patient-grouped cross-validation; the 76
validation scans are a check; the 200 test scans are read **once**, at the end, and
reported against this document whatever they show.

Phase IIIb is the baseline and its lessons are assumed: M0 is the 5-fold ensemble (test
Dice 0.6638, FP 9.10 mL), single fixed actions are pure trades along the existing FROC
curve, a val-tuned constraint did not transfer, and automatic removal costs real lesions
including large ones.

---

## 1. Primary endpoint

**Mean Dice over tumour-positive cases**, agent output versus M0, on test.

- comparison: paired case-level bootstrap, 2000 resamples, 95% percentile interval, over
  the 200 test patients (one scan each, so resampling cases resamples patients);
- the agent output is the mask the loop returns, i.e. the best mask by Q, not the mask at
  a fixed iteration;
- **H3 is supported** iff the interval for ΔDice excludes zero and lies above it.

A positive point estimate with an interval straddling zero is reported as "no effect
demonstrated", not as a trend.

## 2. Safety constraint (gating, fixed in advance)

A configuration is **eligible** only if

> Δ(lesion sensitivity) ≥ −0.01 on the OOF split **and** on val,

lesion sensitivity being the fraction of ground-truth lesion components touched by the
prediction, pooled over tumour-positive cases. This is a non-inferiority margin of one
percentage point, the same margin used in Phase IIIb, and it is evaluated on OOF and val
only. Test sensitivity is reported, never used to select.

If no configuration is eligible, that is the result: the agent is reported as not
deployable and the test row is reported for the ineligible best-on-OOF configuration,
labelled as such.

## 3. Secondary endpoints

Reported on test for M0 and for each iteration M1, M2, M3, with the same paired
bootstrap:

| endpoint | definition |
|---|---|
| FP volume | mean over all cases, mL |
| FN volume | mean over all cases, mL |
| FP lesions / patient | mean count of predicted components with zero GT overlap |
| tumour-free patients with any FP | count out of the 90 tumour-free test cases |
| HD95 | 95th-percentile Hausdorff distance, mm, over cases where both masks are non-empty; the count of undefined cases is reported alongside |
| lesion sensitivity | as in §2 |
| % improved / unchanged / degraded | per §23, with **unchanged** defined as \|ΔDice\| < 0.01; improved is ΔDice ≥ +0.01, degraded ΔDice ≤ −0.01 |

Lesions removed and lesions recovered are additionally tabulated by ground-truth size bin
(< 0.5, 0.5–1, 1–5, 5–20, > 20 mL), as in Phase IIIb.

## 4. Failure classes (definitions fixed here, not after seeing the data)

Computed from ground truth on OOF and val. A predicted component is one connected
component of `P̄ > 0.5` with at least 5 voxels, 6-connectivity, exactly as in
`build_candidates.py`.

| class | definition |
|---|---|
| spurious (FP) | predicted component with zero GT voxel overlap; split by location into physiological (brain, heart, urinary bladder, kidneys), liver/spleen, other, by majority TotalSegmentator label |
| missed (FN) | GT component touched by no predicted component; binned by volume as above |
| under-segmentation | GT lesion that *is* touched, with lesion-level Dice < 0.5 **and** matched predicted volume < 0.6 × GT volume |
| over-segmentation | GT lesion that is touched, with lesion-level Dice < 0.5 **and** matched predicted volume > 1.67 × GT volume |
| fragmentation | one GT lesion touched by ≥ 2 predicted components |
| merging | one predicted component touching ≥ 2 GT lesions |
| boundary error | touched GT lesion with lesion Dice in [0.5, 0.8), and mean predictive entropy in the 2-voxel band around the component surface at least 1.5 × the mean in its interior |

Classes are **not** mutually exclusive; the diagnosis model of eq. 19 is multi-label. The
cost of each class is reported as the Dice recoverable if that class alone were corrected
perfectly, holding the others fixed.

## 5. Diagnosis model P(F_k | O), eq. 19

Multi-label classifier over observable features only — no ground truth: Q and its
component features, uncertainty (`u_ent`, `u_stab`; `u_epi` only where defined, i.e. not
on OOF), SUV statistics, TotalSegmentator organ, size, shape, topology. Fitted on OOF
with patient-grouped 5-fold CV; per-class AUROC reported on OOF (cross-validated), val
and test.

## 6. Strategy library and the per-strategy table

Each of the seven strategies in §10 is applied on OOF to **every component it targets**,
and the effect is tabulated as ΔDice (case), Δlesion sensitivity, ΔFP volume, ΔQ, broken
down by diagnosed failure class. This table is computed on OOF only and is what the
rule-based policy of eq. 30 is read from.

## 7. The agent loop (Algorithm 1) and its free parameters

Per case: compute Q; if Q > τ_Q accept M0 unchanged; else diagnose, apply the strategy
the rule maps the dominant failure to, recompute U and Q, accept iff ΔQ > 0, else reject
and try the next-best strategy for that class; stop when |ΔQ| < ε or after T iterations.
Return the best mask by Q.

Fixed in advance:

- **T = 3** (the plan's T ≤ 3);
- **ε = 0.005** in Q units;
- **τ_Q** tuned on OOF by patient-grouped CV, reported with its per-fold range;
- **λ = 0.001 mm⁻¹** in the experience reward r_t = ΔDice − λ·ΔHD95, so 10 mm of HD95
  trades against 0.01 of Dice;
- strategy order within a failure class is the OOF ranking from §6, ties broken by lower
  ΔFP volume.

## 8. Rule for the final configuration (Rule R4)

A configuration is the tuple (τ_Q, T, strategy ordering). Among configurations **eligible**
under §2 on both OOF and val:

1. maximise mean Dice on OOF (tumour-positive cases);
2. tie-break on lower mean FP volume on OOF;
3. tie-break on fewer mean iterations per case.

Chosen before test is read. One configuration goes to test.

## 9. Q as a judge

Pre-registered diagnostics, OOF and val: the rate at which ΔQ > 0 coincides with
ΔDice > 0; the rate at which ΔQ > 0 coincides with no lesion lost; and, per strategy, the
rate of ΔQ > 0 with ΔDice < 0 ("gaming"). A strategy whose gaming rate on OOF exceeds its
honest rate is removed from the library before test, and this is reported.

## 10. Difficult-case analysis (§22) and H4

The §3 endpoints are additionally split by, all defined on OOF/val quantiles applied to
test without refitting:

- Q tertile (from the independent quality model);
- uncertainty tertile (case mean `u_stab` over the predicted mask);
- inconsistency groups from the §4 diagnosis: **anatomically** inconsistent (any spurious
  component in a physiological-uptake organ), **metabolically** inconsistent (any
  component whose SUVmax is below the PERCIST liver threshold), **topologically**
  inconsistent (any fragmentation or merging).

**H4 is supported** iff the ΔDice point estimate in the lowest-Q tertile exceeds that in
the highest-Q tertile and its bootstrap interval excludes zero.

## 11. Ablation rows (§21)

M0; + uncertainty; + reasoning; + refinement (the loop with the Q gate disabled, every
proposed action accepted); + reflection (the full loop). Reported on test with the same
metrics.

## 12. What would falsify the approach

Stated now so it cannot be renegotiated later:

- H3 fails if the primary ΔDice interval includes zero;
- the agent is not deployable if no configuration is eligible under §2;
- the loop adds nothing over a single pass if M1 and M3 are within 0.005 Dice of each
  other on OOF;
- Q is not a usable judge if ΔQ > 0 coincides with ΔDice > 0 at below 60% on OOF, in
  which case the reflection gate is reported as unusable rather than tuned until it works.
