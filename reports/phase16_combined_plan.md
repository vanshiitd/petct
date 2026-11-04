# Phase XVI: the combined agent -- anchored union of the hand-built policy and the guarded context detector -- pre-registered

Written before any Phase XVI number is computed. Branch `phase16-combined`,
from `phase15-ood`.

## Status and disclosure: a post-terminal attempt

Phase XV declared itself terminal. This phase reopens the question **at the
user's explicit request**: "try again at your level best ... combined all".
That carries a cost, stated plainly.

- **Results that motivated the design.** Every out-of-fold and validation
  result through Phase XV was read before this design was chosen.
  Specifically, the Phase XV out-of-fold Dice shortfall came from
  PETCT_22d07bdc49, a component the hand-built policy acts on.
- **Out-of-fold and validation are descriptive only.** Nothing they show
  here is evidence of a win.
- **Test is the only confirmation.** It has never been read for any
  policy or detector from Phase VII on. It is read **once**, under the lock
  below, and reported whatever it shows.
- **The ledger grows by one phase**, with 24 configurations (the same grid
  as Phases XIV-XV).

## Design: every earlier component, in the role where it worked

| component | from | role here |
|---|---|---|
| hand-built actions are never removed (one-sided trust region) | XII | **anchor**: every component the hand-built detector acts on at 0.90 is acted on with the hand-built class |
| context detector | XIII | proposes **additions** only (components the hand-built policy leaves alone) |
| calibration, old+new ensemble, agreement and solo guards, gate | XIV | the 24 addition configurations: det {`new_cal`, `ens`} x t {0.85, 0.90, 0.95} x guard {none, solo, agree, agree+solo} |
| out-of-distribution abstention (Q3 + 3 IQR fence on component count) | XV | cases above the fence get the hand-built policy only. Out-of-support cases are excluded from calibrator fitting |
| causal-utility promotion (CAVE-PET) | XII | an addition set is promoted only if it passes all of the checks below; otherwise **roll back** to the hand-built policy |

The promotion checks, all on the training folds, are:
- n >= 20;
- patient-clustered 95% LCB of mean `d_qdice` > 0;
- Clopper-Pearson harm upper bound <= 0.05;
- lesions lost <= 2;
- tumour-positive summed `d_qdice` >= -0.5.

The resulting policy, per component:

```
act = hb_act  OR  ( in_support AND NOT hb_act AND config_act )
class = hand-built argmax   if hb_act
        config argmax       otherwise
```

Every constant comes from an earlier committed phase, and none is new:
- the promotion thresholds are Phase XII's (`phase12_anchored.py`);
- the fence and calibration are Phase XV's;
- the grid is Phase XIV's.

**Selection on T.** Among promotable configurations, take the largest summed
`d_qdice` of the added set. Ties go to `ens`, then to the higher `t`, then to
the stronger guard.

**Nesting.** Nesting is identical to Phase XV:
- stage-1 heads are reproduced exactly;
- calibrators are fitted on inner cross-fits within T;
- the fence comes from T.

Production is selected the same way on all out-of-fold data.

## Deployment

Through `--pfail-table`:
- hand-built rows keep the old raw probabilities;
- added rows carry the configuration's probabilities;
- all other rows are zero;
- recovery rows are copied as before.

The agent runs with `--min-pfail 1e-6 --policy phase4_policy_noban.json --T 10`.

## Endpoints

These are unchanged from Phases XIII-XV, paired against the hand-built run.

There are two superiority routes, so each is tested at a Bonferroni level
of two-sided 0.025.

1. **FP-clearance superiority.** McNemar p < 0.025.
2. **Dice superiority.** dDice > 0, with bootstrap p < 0.025.
3. **Dice non-inferiority.** Lower 95% bound >= -0.0025.
4. **Lesion non-inferiority.** Lower 95% bound >= -5.
5. **Beats the hand-built policy overall.** (1 or 2) and 3 and 4.

**Validation safety check, required for test:**
- dDice >= -0.0025;
- lesion hits retained >= -1;
- no new false positives.

**Test.** Test is read once, whatever the out-of-fold result, if:
- production promotes an addition set, and
- validation passes the safety check.

It is run for this policy and for the hand-built test table
(`phase11_eag_hb_test.npz`), after a lock commit that records:
- SHA-256 of `phase16_comb_test.npz`, `phase11_eag_hb_test.npz` and
  `cand_cv_test_pos.csv`;
- both commands.

**On test, criterion 5 is the confirmatory result.**

If production rolls back, the deployed policy is the hand-built policy, and
test is not run. That case is reported as "no learned change deployed".

## Descriptives

- Per-fold selection with the promotion table.
- Additions held out: n, summed `d_qdice`, lesions lost.
- Abstaining cases.
- The subgroup table.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
