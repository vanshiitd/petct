# Phase XXI result: selection rules, shift stress test, safeguarded switch

Plan `reports/phase21_baselines_shift_safeguard_plan.md` (f4f986b); lock
`reports/phase21_lock.md` (8ea44df). Everything here is **post hoc**: B and C
had been read before the plan was written. Nothing here is a certificate.

## Part A: selection rules in the validity study

1,000 random halves of B ∪ C (125 select, 126 evaluate). The CSI and naive
rows reproduce `phase19_eval_psma.json` to 1e-12. The harm indicator is
monotone along every chain (0 of 20,080 gate steps reverse).

Violation = share of splits whose evaluation harm rate exceeds δ; benefit =
mean summed Dice gain on the evaluation half.

| rule | δ 0.05: violation [95% CI] | benefit | δ 0.10: violation [95% CI] | benefit |
|---|---|---|---|---|
| NAIVE | 0.985 [0.975, 0.992] | +5.95 | 0.350 [0.320, 0.380] | +5.95 |
| PLUGIN | 0.409 [0.378, 0.440] | +4.52 | 0.350 [0.320, 0.380] | +5.72 |
| CRC | 0.253 [0.226, 0.281] | +3.98 | 0.350 [0.320, 0.380] | +5.59 |
| UNCORR | 0.071 [0.056, 0.089] | +2.40 | 0.017 [0.010, 0.027] | +4.52 |
| ZERO | 0.005 [0.002, 0.012] | +0.83 | 0.000 [0.000, 0.004] | +0.83 |
| BONF-88 | rolls back in every split | 0 | 0.002 [0.000, 0.007] | +2.40 |
| CSI | 0.005 [0.002, 0.012] | +0.83 | 0.004 [0.001, 0.010] | +3.50 |

Against the plan's expectations:
- **A1 holds.** CSI is within α at both δ.
- **A2 holds.** NAIVE, PLUGIN and UNCORR exceed α at δ = 0.05.
- **A3 holds.** BONF-88 gives less benefit than CSI at both δ.
- **A4 (no prediction).** CRC keeps its own promise on the mean (harm rate
  0.039 at δ 0.05, 0.077 at δ 0.10) but exceeds δ in 25% and 35% of splits.
  ZERO picks the same candidate as CSI in every split at δ = 0.05, because
  with n = 125 and α/8 only k = 0 certifies. ZERO cannot use a larger δ
  (benefit +0.83 throughout) and violates in 13.6% of splits at δ = 0.03,
  where CSI rolls back.

**Against the paper (pre-stated criterion).** At δ = 0.10, UNCORR has a
violation rate within α (0.017) and more benefit than CSI (+4.52 against
+3.50). In this study it dominates CSI at that δ. It exceeds α at δ = 0.05
(0.071) and δ = 0.03 (0.136), so it does not hold its level across δ.

## Part B: certification under tracer or vendor shift

Select on half of the source group; control = the other half; shifted = the
whole target group. 1,000 draws, δ = 0.10. Violation rates:

| direction (select / shifted n) | NAIVE control | NAIVE shifted | CRC control | CRC shifted | CSI control | CSI shifted |
|---|---|---|---|---|---|---|
| 18F → 68Ga (71 / 109) | 0.115 | 1.000 | 0.115 | 0.749 | 0.012 | 0.009 |
| 68Ga → 18F (54 / 142) | 0.513 | 0.000 | 0.171 | 0.000 | 0.013 | 0.000 |
| vendor 1 → 2 (74 / 103) | 0.085 | 1.000 | 0.085 | 0.734 | 0.004 | 0.004 |
| vendor 2 → 1 (51 / 148) | 0.631 | 0.000 | 0.191 | 0.000 | 0.022 | 0.000 |

- CSI's selected update stayed within δ on the shifted group in at least
  99.1% of draws in every direction. This is an observation on two shifts
  inside one dataset, not a guarantee.
- The shift is asymmetric: the 68Ga and vendor-2 groups are harder. Naive
  selection on the easier group looks acceptable on its control and exceeds
  δ on the harder group in every draw.
- At δ = 0.05 the selection halves (51 to 74 patients) are too small to
  certify anything, and CSI rolls back in every draw, as the plan stated.

## Part C: safeguarded switch — negative result

S_τ = FT ∪ (P19 ∩ {p_FT ≥ τ}). All three reproduction checks passed on B
and on C.

| τ | B harmed /126 | B Dice | B FP mL | C harmed /125 (UB) | C Dice | C FP mL | C hits |
|---|---|---|---|---|---|---|---|
| 0 (FT ∪ P19) | 0 | 0.1636 | 635.1 | 1 (3.7%) | 0.1642 | 598.8 | 3584 |
| 0.01 | 4 | 0.5692 | 5.8 | 2 (5.0%) | 0.5424 | 16.9 | 3563 |
| 0.02 | 4 | 0.5700 | 5.7 | 2 (5.0%) | 0.5436 | 16.6 | 3560 |
| 0.05 | 4 | 0.5709 | 5.6 | 2 (5.0%) | 0.5448 | 16.1 | 3559 |
| 0.10 | 4 | 0.5713 | 5.4 | 3 (6.1%) | 0.5457 | 15.8 | 3556 |
| 0.20 | 4 | 0.5722 | 5.0 | 3 (6.1%) | 0.5470 | 15.5 | 3556 |
| 0.30 | 4 | 0.5722 | 4.7 | 3 (6.1%) | 0.5469 | 15.1 | 3555 |
| 0.40 | 4 | 0.5722 | 4.4 | 3 (6.1%) | 0.5470 | 14.8 | 3550 |
| 0.50 (FT) | 4 | 0.5714 | 4.2 | 3 (6.1%) | 0.5470 | 14.5 | 3547 |
| P19 | | 0.1216 | 632.3 | | 0.1158 | 592.4 | 2094 |

- **The safeguard fails.** On B, no τ above 0 removes any of FT's four
  harms. FT assigns the lesions it loses a probability below 0.01 inside the
  deployed mask: it is confidently wrong there, not unsure.
- Selected by the pre-registered rule: τ* = 0 at δ = 0.05 and τ* = 0.20 at
  δ ≥ 0.10.
  - τ* = 0 on C: 1 of 125 harmed, mean Dice 0.164 (−0.383 against FT). It
    keeps the deployed false positives, so it gives up almost all of FT's
    gain. The one harmed patient loses Dice, not a lesion.
  - τ* = 0.20 on C: 3 of 125 harmed, Dice 0.5470, the same as FT.
- On C, τ ≤ 0.05 has one harm fewer than FT (2 against 3). The rule did not
  select these τ, and B shows no such effect, so nothing is claimed from it.

**Conclusion.** Falling back to the old output where the new model is unsure
does not make this switch safer, because the new model is not unsure where
it fails. Confidence cannot stand in for a harm test. The paper reports this
as a negative result.

## Files

`D:\data\reports`: `phase21_rules.json`, `phase21_rules.log`,
`phase21_safeswitch_B.json`, `phase21_safeswitch_C.json`, their logs,
`p21{B,C}_psma_p19_*.csv`, `p21_masks_{B,C}/`.
