# Phase IX: certified class-specific gates -- pre-registered

Written before any cross-fitted number in this document is computed.
Branch `phase9-class-gates`, from `phase8-causal-override` (which carries
Phase VII and its FP-cleared correction).

## What the earlier phases actually showed, once compared head-to-head

Every verdict in Phases VII and VIII compared a policy's gain over M0
against the hand-built policy's gain over M0 -- two separate intervals.
Comparing each run to the hand-built policy's **own run, case by case**
(paired, same 353 tumour-positive OOF cases) shows where the gap comes from:

| run vs. hand-built (b), paired | mean dDice | cases that differ at all |
|---|---|---|
| (a) fixed `topology_remove` | -0.0032 | 11 |
| piece 2 (pairwise ranking) | -0.0032 | 11 |
| Phase VIII (causal override) | -0.0033 | 15 |

**Essentially the whole gap is two or three patients.** For (a) and piece 2
it is two cases where `topology_remove` was applied to a component diagnosed
`merging` -- a real tumour -- costing -0.62 and -0.52 Dice; the hand-built
policy only allows `topology_fill` for `merging` and never does this. For
Phase VIII it is two cases (`PETCT_978c395243`, `PETCT_22d07bdc49`) where a
single large false-positive blob is worth +0.69 and +0.38 Dice when deleted;
the hand-built policy deletes it, Phase VIII's override sent exactly that
blob to `roi_repredict` (+0.017).

So the lessons are: the large gains come from **deleting large false
positives**; the large losses come from **deleting real tumour**; and the
strategy choice has almost no headroom left -- the hand-built policy's picks
are already right. What has not been tried is changing **which components
are acted on at all**, safely.

## Where the headroom is

Using the hand-built policy's own pick per component (in-sample p_failure;
this phase recomputes everything with the cross-fitted p_failure from
`audit_crossfit.npz`):

| `spurious_other`, p_failure band | n | sum `d_qdice` | harmful outcomes |
|---|---|---|---|
| [0.95, 1.0] (gated today) | 471 | +22.54 | 0 |
| [0.90, 0.95) (gated today) | 101 | +2.57 | 1 |
| **[0.80, 0.90) -- not gated** | 316 | **+11.60** | **0** |
| [0.70, 0.80) -- not gated | 423 | +11.24 | 5 |
| [0.50, 0.70) | 985 | +12.03 | 54 |
| [0.00, 0.50) | 1747 | -16.17 | 223 |

Below 0.5 is where piece 4's global gate removal did its damage. Between 0.7
and 0.9 there is almost as much benefit again as the whole gated region
provides, with very little measured harm. No other class shows comparable
headroom below the gate.

## The mechanism

The hand-built policy's strategy choice is kept **exactly**. Only the gate
changes, per failure class, one p_failure slice at a time:

slices: [0.85, 0.90), [0.80, 0.85), [0.75, 0.80), [0.70, 0.75),
[0.65, 0.70), [0.60, 0.65), [0.50, 0.60).

On the four training folds of each outer fold (3,4,5,6,7), using
**cross-fitted** p_failure and class, a class's gate drops through a slice
only if that slice, on its own:

1. has at least 30 components;
2. **benefit**: the patient-clustered bootstrap (2000 resamples) lower 95%
   bound of mean `d_qdice` of the hand-built pick is above zero;
3. **safety**: the one-sided 95% upper bound (Clopper-Pearson) of the harm
   rate is at most 2%, where harm is `lesions_lost > 0` or (tumour-positive
   case and `d_dice < -0.01`).

The gate stops at the first slice that fails, so a strong upper slice cannot
carry a harmful lower one. The safety condition is a different metric from
the one optimised, which is exactly what Phase VIII's guard lacked.

Deployment: `phase5_agent.py --min-pfail <lowest promoted gate>` with an
`--avpkl` wrapper that returns the hand-built score
(`len(order) - order.index(strat)`, identical to the native branch) for a
candidate whose p_failure clears its class's gate and -1 otherwise. With
every gate at 0.90 the wrapper is behaviourally identical to the native
hand-built run; this is checked directly -- every case where no newly
gated component exists must reproduce (b)'s per-case Dice exactly.

## Endpoints and falsification

- **Primary (new, correct comparison)**: paired bootstrap over the 353 OOF
  tumour-positive cases of (this run - hand-built run), 2000 resamples.
  **Supported** if the interval excludes zero on the positive side.
- **Co-primary safety**: lesion sensitivity change against M0 within the
  -0.01 budget, and no more degraded cases than the hand-built run.
- Also reported: dDice vs. M0, tumour-free patients cleared (the corrected
  `mcnemar_fp` statistic), validation as a check.
- **Per-slice**: every promoted slice's held-out-fold mean `d_qdice` must
  agree in sign with the training folds that promoted it.

Test untouched unless the configuration survives OOF and validation. No
prior-phase file modified.
