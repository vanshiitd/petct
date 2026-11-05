# Phase XVII result: the first certified learned update -- safe on the locked test, but its gain is too small to beat the hand-built policy

Scripts:
- `scripts/phase17_ltt.py`
- `scripts/phase17b_lifelong.py`

Plans:
- `reports/phase17_ltt_plan.md` (f18c200)
- `reports/phase17b_lifelong_validity_plan.md` (0f07cad, corrected in
  4bccda5)
- test lock `reports/phase17_test_lock.md` (9ac9149)

Outputs are in `D:\data\reports\`:
- `phase17_ltt.json`
- `phase17_ltt_{oof,val,test}.npz`
- `phase17_eval_{val,test}.json`
- `phase17b_lifelong.json`
- `p17_{val,test}_*`

## Checks

The heads reproduce `audit_crossfit.npz` and `phase13_det_oof.npz`
exactly.

## 1. Certification (out-of-fold, 630 patients, δ = 0.01, α = 0.05)

Learn-then-Test certified **36 of 88** candidates. The `ens` chains
certify down to gate 0.93. The `new_cal` chains certify down to 0.98.

**Deployed:** `new_cal`, gate 0.98, guard none. It is the first learned
update promoted in this project.

| added | patients harmed | p-value | lesions lost | summed d_qdice | benefit LCB | tumour-free / tumour-positive part |
|---|---|---|---|---|---|---|
| 48 | **0** | 0.0018 | **0** | +4.47 | +0.023 | +4.00 / +0.47 |

**Risk-benefit curve** (descriptive). δ is the tolerated fraction of patients
harmed:

| δ | certified | deployed | summed d_qdice | patients harmed |
|---|---|---|---|---|
| 0.005 | 0 | rollback | 0 | 0 |
| **0.01** | 36 | `new_cal` 0.98 | **+4.47** | **0** |
| 0.02 | 62 | `new_cal` 0.95 | +13.63 | 4 |
| 0.03 | 66 | `new_cal` 0.94 agree | +16.77 | 6 |
| 0.05 | 87 | `new_cal` 0.85 agree | +25.38 | 18 |

At α = 0.10 the curve is identical, except at δ = 0.03: `new_cal` 0.93
agree, +19.82, k = 9.

Benefit grows about 6× from δ = 0.01 to δ = 0.05, and patients harmed go
from 0 to 18. The tolerance is now an explicit, certified dial instead of
an ad hoc count.

## 2. Validation safety check: pass

- Production adds 1 component.
- Tumour-free cleared: 2 vs 2.
- dDice +0.0000.
- Lesion hits retained +0.
- New false positives: 0 vs 0.

## 3. Locked test (200 scans, read once)

| | hand-built | Phase XVII |
|---|---|---|
| Dice (aggregate) | 0.4606 | **0.4635** |
| FP volume (mL, mean) | 7.770 | **7.691** |
| FN volume (mL, mean) | 5.292 | 5.292 |
| lesion hits | 985 | 985 |
| tumour-free cleared | 6 / 90 | **7 / 90** |

Paired endpoints:

| criterion | result | |
|---|---|---|
| 1 FP superiority | 7 vs 6; discordant 1 vs 0; McNemar p = 1 | fail |
| 2 Dice superiority | +0.0000 per case, p = 0.95 | fail |
| 3 Dice NI | +0.0000 [-0.0000, +0.0000] | pass |
| 4 lesion NI | +0 [+0, +0] | pass |
| **5 beats the hand-built policy** | | **no** |

**Changed scans: 5 of 200.**
- 4 improve:
  - PETCT_4e3d83bfe3, FP 15.2 -> 0.2 mL;
  - PETCT_d098458d29, cleared;
  - PETCT_d472033d3b, FP 1.8 -> 1.1 mL;
  - PETCT_9837205b34, FP -0.1 mL.
- 1 has a negligible Dice change: PETCT_98c6af8b90, -0.0003.

**Patients harmed: 0 of 200.** The exact one-sided 95% upper bound is
0.015. That is consistent with δ = 0.01, though it cannot confirm it at
this n.

**Reading.** The guarantee behaved as promised: the certified update did no
harm on unseen data. But at δ = 0.01 the update is so conservative that it
touches only 5 test scans. Its gain is real in direction but far too small
for significance, and the confirmatory endpoint 5 is not met.

## 4. Validity of the guarantee

Setup: 1000 random 50/50 patient splits, with certification on 315
patients and evaluation on 315.

| δ | LTT: P(eval harm > δ) | LTT: mean harm | LTT: benefit | naive: P(eval harm > δ) | naive: mean harm | naive: benefit |
|---|---|---|---|---|---|---|
| 0.02 | **0.008** | 0.003 | +3.04 | **0.766** | 0.028 | +12.02 |
| 0.03 | 0.010 | 0.009 | +7.09 | 0.402 | 0.028 | +12.02 |
| 0.05 | 0.007 | 0.020 | +11.22 | 0.007 | 0.028 | +12.02 |

The naive selector picks the best-looking update on the calibration half.
**It violates a 2% tolerance in 77% of splits.** Certified selection violates
it in under 1% of splits, well within α = 5%.

The benefit cost of certification shrinks as δ grows: at δ = 0.05, LTT
keeps 93% of the naive benefit with no added violations.

The evaluation half is finite, so violation frequency is a noisy,
conservative proxy for P(R > δ).

## 5. Lifelong (two-round) certification

| round | train | certify (patients) | α_r | δ = 0.02 | δ = 0.03 | δ = 0.05 |
|---|---|---|---|---|---|---|
| 1 | folds 3-4 | 5-7 (377) | 0.025 | `ens` 0.85 agree: +5.32, k = 0 | `new_cal` 0.94 agree: +6.25, k = 1 | same as δ = 0.03 |
| 2 | folds 3-5 | 6-7 (251) | 0.0125 | rollback | rollback (44 certified, none passes the benefit check) | `new_cal` 0.93 agree: +5.23, k = 2 |

Round 1 certifies a safe update at every δ. Round 2, retrained on more
data, does **not** show a larger benefit. It has fewer fresh patients
under a halved α_r, so it rolls back at δ <= 0.03.

The self-improvement-across-rounds claim is **not supported** by this
simulation. What it does show is the price of a lifelong guarantee: about
300 or more fresh patients per round at these tolerances. The k = 0 bound
is n >= ln(α_r / 8) / ln(1 - δ).

## What this says

1. **Positive and confirmed:** a learned update was certified and deployed.
   On a never-read test set it harmed no patient, as the guarantee
   promised.
2. **Positive, strong:** certification is what separates safe from unsafe
   self-improvement. Naive selection violates the tolerance up to 77% of
   the time; LTT does so under 1% of the time.
3. **Negative:** at the safe level δ = 0.01, the certified gain on test is
   too small to beat the hand-built policy (endpoint 5 fails).
4. **Negative:** lifelong rounds do not yet show growing benefit; fresh
   data per round is the bottleneck.

The test split has now been read for Phase XVII. It cannot be reused to
confirm any later choice.

## Deviations

- The test runs were chained in one shell command. Its exit code 1 came
  from the trailing `tail` over two files. Both agent runs completed
  normally (200/200 cases each), and each was made once.
