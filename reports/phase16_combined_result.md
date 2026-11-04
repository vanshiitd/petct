# Phase XVI result (post-terminal): the combined agent's additions cannot be certified -- every fold and production roll back

Script: `scripts/phase16_combined.py`.

Outputs, written to `D:\data\reports\`:
- `phase16_comb.json`
- `phase16_comb_{oof,val,test}.npz`

Pre-registered in `reports/phase16_combined_plan.md` (commit 2e411b2).

## Checks

The heads reproduce `audit_crossfit.npz` and `phase13_det_oof.npz` exactly.

## Result

**No configuration was promotable on any outer fold or in production. The
deployed policy is the hand-built policy.**

| split | acted components | hand-built acted |
|---|---|---|
| out-of-fold | 630 | 630 |
| validation | 26 | 26 |
| test | 94 | 94 |

Out-of-fold, validation and test match the hand-built policy exactly, and
no addition is deployed.

Under the plan, that is "no learned change deployed". The agent was not
re-run, because the table equals the hand-built table, and test was not
run. Test remains unread for every policy and detector from Phase VII on.

## Why: the lesion limit, and only the lesion limit

These are production promotion statistics: additions only, all out-of-fold
data, the top configurations by summed `d_qdice`.

| config | n | summed d_qdice | utility LCB | harm UB (<= 0.05) | lesions lost (<= 2) | tumour-pos sum (>= -0.5) |
|---|---|---|---|---|---|---|
| `new_cal` 0.85 agree | 386 | +25.4 | +0.041 | 0.071 x | 19 x | -0.62 x |
| `new_cal` 0.90 agree | 349 | +24.6 | +0.045 | 0.052 x | 11 x | -0.38 |
| `new_cal` 0.95 none | 186 | +13.6 | +0.039 | 0.049 | **4 x** | +0.63 |
| `new_cal` 0.95 agree | 178 | +13.6 | +0.039 | 0.051 x | 4 x | +0.58 |
| `ens` 0.90 none | 80 | +3.8 | +0.005 | 0.077 x | 2 | -0.17 |

Every configuration has a positive causal-utility lower bound: the additions
help on average, with confidence. The safest productive one,
`new_cal` 0.95, passes every criterion except one. It deletes 4 true lesions
among 186 additions over 738 scans, against a limit of 2.

## What this says

Phase XVI combines every mechanism that worked in Phases XII-XV:
- anchoring to the hand-built actions;
- the context detector;
- calibration, the ensemble and the guards;
- out-of-distribution abstention;
- causal promotion.

It shows the trade-off without ambiguity:

- **Every extra clean scan the learned detector buys costs true lesions at a
  low but nonzero rate.** About 1 lesion per 45 additions at the strictest
  gate.
- **Under a safety rule that tolerates almost no extra lesion loss** (<= 2
  across the whole training set), **no learned addition can be certified**.

Any rule loose enough to deploy one would be chosen after seeing this
table, and would be a relaxation of safety for the sake of a win. This report
does not do that.

The hand-built policy therefore remains the deployed policy. The learned
system's contribution is the **certified knowledge** of where its gains are
and what they cost. It stays deferential where it cannot certify itself.

## Deviations

None.
