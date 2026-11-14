# Phase XXI lock: Part C, before S_tau is computed on C

Plan: `reports/phase21_baselines_shift_safeguard_plan.md` (f4f986b). This
file is committed before `phase21_safeswitch.py --stage C` runs.

## Known before the lock (B only)

Checks passed on B: the regenerated P19 table equals the locked one (126
cases); the saved P19 masks reproduce its Dice and hits; S_0.5 reproduces the
locked FT rows.

| τ | harmed of 126 | of which lost a hit | mean Dice (tumour-positive) | FP mL | hits |
|---|---|---|---|---|---|
| 0 (FT ∪ P19) | 0 | 0 | 0.1636 | 635.1 | 3353 |
| 0.01 | 4 | 3 | 0.5692 | 5.8 | 3327 |
| 0.02 | 4 | 3 | 0.5700 | 5.7 | 3325 |
| 0.05 | 4 | 3 | 0.5709 | 5.6 | 3324 |
| 0.10 | 4 | 3 | 0.5713 | 5.4 | 3321 |
| 0.20 | 4 | 3 | 0.5722 | 5.0 | 3319 |
| 0.30 | 4 | 3 | 0.5722 | 4.7 | 3316 |
| 0.40 | 4 | 3 | 0.5722 | 4.4 | 3311 |
| 0.50 (FT) | 4 | 3 | 0.5714 | 4.2 | 3306 |
| P19 | | | 0.1216 | 632.3 | 2038 |

**The safeguard does not do what it was designed for.** No τ above 0 removes
any of FT's four harms on B: FT gives the lesions it loses a probability
below 0.01 inside the deployed mask. Only the full union (τ = 0) is
harm-free, and it keeps the deployed system's false positives.

Selection by the pre-registered rule:

| δ | τ that pass | τ* |
|---|---|---|
| 0.05 | 0 | 0 (FT ∪ P19) |
| 0.10, 0.15, 0.20 | all | 0.20 |

τ* = 0 at δ = 0.05 follows the rule (its mean Dice, 0.1636, exceeds P19's
0.1216), but it is the degenerate end of the family, not a useful switch.
This is reported as a negative result. The rule is not changed.

**Deviation from the plan's wording.** The plan writes {p_FT ≥ 0.5}; the code
uses {p_FT > 0.5}, the locked FT definition, so that S_0.5 equals FT exactly.

## SHA-256 (`D:\data\reports`)

```
f87f713cb3a1506fd0bca4a88d5918d628d0c47eea5816fb9dc5b62d933ca411 *phase21_safeswitch_B.json
0a931e3337bffaca6a7eb3809df737919606cddb8a10401e86f0ed473f93a295 *phase21_rules.json
ae3040dcd1301a93d0c602b32c58eafefca7342923e104c72f1ebdca54904407 *p21B_psma_p19_final.csv
f4b40371cb6d24d86bd38c01ea0287a7afd96a19c94e8a93eedd35b3c007e998 *p21C_psma_p19_final.csv
b95c930fb00ec17517b3cdba25f97b62f6057252d9b79c5b3ebf2b052d7b841e *phase19_psma_p19_C.npz
```

## Command (once)

```
../petct/.venv/Scripts/python.exe ../petct/scripts/phase21_safeswitch.py --stage C
```
