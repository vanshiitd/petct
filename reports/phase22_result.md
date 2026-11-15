# Phase XXII result: lowering the operating point does not make the switch safer

Plan `reports/phase22_operating_point_plan.md` (5b79b5f). **Failure by the
plan's criterion: no threshold passes on B at δ = 0.05. C was not opened.**

## B (n = 126), FT_t = {p_FT > t} against P19

Checks passed: the P19 masks and FT_0.5 reproduce the locked rows.

| t | harmed | of which lost a hit | mean Dice | FP mL | hits | Dice vs FT_0.5 [95% CI] |
|---|---|---|---|---|---|---|
| 0.02 | 4 | 3 | 0.5320 | 15.5 | 3569 | −0.0395 [−0.0605, −0.0203] |
| 0.05 | 4 | 3 | 0.5393 | 13.7 | 3538 | −0.0321 [−0.0519, −0.0143] |
| 0.10 | 4 | 3 | 0.5449 | 12.1 | 3502 | −0.0265 [−0.0450, −0.0095] |
| 0.20 | 4 | 3 | 0.5597 | 8.9 | 3448 | −0.0117 [−0.0251, +0.0003] |
| 0.30 | 4 | 3 | 0.5657 | 6.8 | 3406 | −0.0057 [−0.0154, +0.0030] |
| 0.40 | 4 | 3 | 0.5714 | 5.2 | 3357 | −0.0000 [−0.0035, +0.0032] |
| 0.50 (FT) | 4 | 3 | 0.5714 | 4.2 | 3306 | 0 |
| P19 | | | 0.1216 | 632.3 | 2038 | |

- The same four patients are harmed at every threshold, down to 0.02.
  Lowering the threshold finds 263 more lesions overall but not the ones
  these patients lose.
- Unlike on A, Dice falls as the threshold falls (−0.04 at t = 0.02). The
  ensemble used on B is sharper than the single out-of-fold models on A.
- Selection: rollback at δ = 0.05 at both error levels; t* = 0.50, plain FT,
  at δ ≥ 0.10. Phase XX's outcome is unchanged.

## What the three attempts show together

| attempt | where it was designed | result on B |
|---|---|---|
| Fall back to the deployed mask where FT is unsure (XXI-C) | from B's known harms | 4 harmed at every τ > 0 |
| Per-patient gate on disagreement features (XXII, dev) | A | no signal on A; abandoned |
| Lower operating point (XXII) | A | 4 harmed at every t |

The four B harms do not come from a threshold or from uncertainty. The
fine-tuned ensemble assigns those lesions a probability near zero. No
training-free change to its output recovers them without re-importing the
deployed system's false positives (the full union, Dice 0.16).

With FT's observed harm rate (7 of 251 on B ∪ C, 2.8%), a 5% certificate
needs either a model that fails less often, which means retraining, or
about 200 certification patients (Phase XX, Table S2).

A, where the idea looked promising, had only two harmed patients. That it
did not transfer is the expected fate of a direction fitted to two events.
