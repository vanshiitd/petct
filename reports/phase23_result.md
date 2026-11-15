# Phase XXIII result: both certificates hold on 143 never-read follow-up scans

Plan `reports/phase23_followup_confirmation_plan.md` (69c66f6); tooling
93344b7; lock 9fe88d1. Agents run once each, evaluation once. Full output:
`reports/phase23_eval.json` (copy of `D:\data\reports\phase23_eval.json`).

Cohort: 143 follow-up studies of 111 B/C patients, none read before this
phase; no exclusions. These are later studies of patients whose earliest scan
certified (B) or tested (C) the systems: a **temporal-shift confirmation**,
not new patients.

## Primary confirmations: both met

| | certificate | harmed patients | exact 95% UB | p | rejects iff | result |
|---|---|---|---|---|---|---|
| **H1** | Level 2: FT vs P19, δ = 0.10 | **4 / 111** (7 / 143 studies) | 8.1% | 0.011 | k ≤ 5 | **met** |
| **H2** | Level 1: P19 vs HB, δ = 0.05 | **0 / 111** (0 / 143 studies) | 2.7% | 0.0034 | k ≤ 1 | **met** |

Holm over H1 and H2 (family-wise 0.05): both met.

## Benefit (patient-cluster bootstrap, 105 patients with a tumour-positive follow-up)

| | ΔDice [95% CI] | one-sided p |
|---|---|---|
| **FT − P19 (Level-2 gain)** | **+0.411 [+0.364, +0.457]** | < 0.001 |
| FT − M0 | +0.416 [+0.367, +0.463] | < 0.001 |
| P19 − HB (Level-1 gain) | +0.0045 [+0.0007, +0.0109] | < 0.001 |
| P19 − M0 | +0.0059 [+0.0018, +0.0123] | < 0.001 |

| arm | Dice per tumour-positive study | FP mL / study | lesions hit (of 6,290) | negatives cleared (of 21) |
|---|---|---|---|---|
| M0 | 0.154 | 716.9 | 2,805 | 0 |
| HB | 0.155 | 710.0 | 2,800 | 0 |
| P19 | 0.160 | 623.5 | 2,800 | 0 |
| FT | **0.580** | **16.2** | **5,022** | 5 |

The Level-1 gain is about the size it was on C (+0.0068), and the Level-2
gain is larger than on C (+0.44 then, on the earliest scans; Dice 0.58 now
against 0.55).

## Secondary

- **FT vs P19 at δ = 0.05:** 4 of 111, p = 0.34. Not met, consistent with B:
  the switch's harm rate is near 4%, so a 5% promise is not supported by
  about 110 patients. The 95% UB is 8.1%.
- **Subsets** (not powered; thresholds become k ≤ 1 or impossible):
  - C patients: H1 1 of 54 (met); H2 0 of 54.
  - B patients: H1 3 of 57; H2 0 of 57.

## The harmed follow-ups

| patient | studies harmed | per study (hits P19→FT, Dice P19→FT) |
|---|---|---|
| 4ee29f… (B) | 4 of 4 | 4→3 (0.02→0.64); 3→2 (0.03→0.32); 2→1 (0.02→0.33); 3→2 (0.03→0.50) |
| 652cdc… | 1 | 2→0 (0.02→0.00) |
| 8effd7… (B) | 1 | 14→13 (0.76→0.23) |
| daf987… (B) | 1 | hits 7→17, Dice 0.69→0.04 |

**Three of the four harmed patients are the same B patients FT harmed at
certification** (Phase XX lock: 4ee29f, 8effd7, daf987). Harm under the
switch is a property of the patient and persists over time. This is why the
patient, not the study, is the right unit; counted per study, the rate would
be 7 of 143 (4.9%) with correlated events. For patient 4ee29f the switch
loses one lesion per study while raising Dice from about 0.03 to 0.3–0.6,
the trade-off the harm definition is designed to flag.

## What this does and does not show

- Both certificates, issued on B before these scans existed in any analysis,
  held on scans no model or person had read, with the pre-registered
  thresholds.
- The patients are not new and the scans are later in time, so this is
  confirmation under temporal shift. Theorem 1 does not cover it.
- The 5% Level-2 promise is still not supported, here as on B.
