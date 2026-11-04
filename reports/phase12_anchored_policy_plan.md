# Phase XII: KL-anchored policy improvement over the hand-built policy -- pre-registered

Written before any Phase XII configuration is fitted or run, and before any
validation or test number for it is computed. Branch `phase12-anchored`, from
`phase11-trust`.

## Disclosure

Out-of-fold and validation results from Phase XI motivated this design and
were read before this plan was written:

- the experience-trained gate cleared 20 tumour-free patients the
  hand-built policy did not, and 3 the other way, with no new false positives;
- its Dice deficit came from components *inside* the 0.90 gate that it
  declined to act on, where the hand-built policy did act. PETCT_22d07bdc49
  alone accounts for -0.0013 of -0.0017.

**Out-of-fold and validation results here are therefore not confirmatory. Test is.**

## Method, in RL-for-LLM terms

| RLHF / RLVR concept | here |
|---|---|
| reference policy pi_ref (the SFT model) | the hand-built policy at the 0.90 gate |
| verifiable reward | measured outcome of an action against ground truth (`d_qdice`, `lesions_lost`) |
| reward model / verifier | the Phase XI harm model (case context, cross-fitted) |
| KL / trust-region anchoring (PPO clip, KL penalty) | **pi may add actions to pi_ref but never remove one**; additions only where the verifier is confident |
| rejection sampling with a verifier | every proposed addition is screened by predicted harm <= h |
| iterative promotion with held-out check | promote/rollback controller below |

The learned policy is

```
act(k) = [ pmax_k >= 0.90 ]                              # pi_ref, always kept
         OR [ g <= pmax_k < 0.90  AND  harm_k <= h ]      # verified additions
```

The strategy choice for each acted component is the hand-built policy's.
`harm_k` is exactly Phase XI's model, with the same nesting:
- stage 1 is nested inside the training folds;
- the harm model is cross-fitted inside the training folds for selection, and
  fitted on all training folds to score the held-out fold.

**Why an asymmetric trust region.** Phase X showed that removing reference
actions trades Dice for lesions at the frontier rate. Phase XI showed that
adding actions where the verifier is confident is where the off-frontier
gain lies. The anchor guarantees that every pi_ref action survives, so any
change is attributable to the additions.

## Causal promotion of additions (CAVE-PET sections 5.2-5.3)

On the training folds of each outer fold, a configuration (`g`, `h`), from
`g` in {0.80, 0.70, 0.60, 0.50} and `h` in {0.01, 0.02, 0.05, 0.10}, is
**promotable** only if its *added* set (components acted on only because of
the addition) satisfies **all** of:

1. at least 20 added components;
2. mean measured `d_qdice` of the added set has a patient-clustered bootstrap
   95% lower bound > 0, which is the causal utility LCB;
3. harm rate in the added set has a one-sided 95% Clopper-Pearson upper bound
   <= 0.05;
4. lesions lost by the added set <= 2;
5. **protected subgroups.** On tumour-positive patients, the added set's
   summed `d_qdice` is >= -0.5. This is the subgroup no-regression condition.

Among promotable configurations, choose the one with the largest summed
`d_qdice` of the added set. Ties go to the larger `g`, then the smaller `h`.
**If none is promotable, roll back to pi_ref.** The production configuration
is selected the same way on all out-of-fold data.

## Deployment

The decisions go into a new `--pfail-table` npz:
- acted rows keep their stage-1 probabilities;
- other rows are zeroed;
- recovery probabilities below 0.90 are zeroed.

The agent runs with `--min-pfail 1e-6`. Every other flag matches the
hand-built reference: `--policy phase4_policy_noban.json --T 10`, no Q gate,
no ROI. `phase5_agent.py` is unchanged. The route was verified exact in
Phase XI: 10 of 10 out-of-fold cases and 76 of 76 validation cases.

## Endpoints

All are paired, case by case, against the hand-built run (`au_{split}_attr_policy`).

1. **FP-clearance superiority.** Among tumour-free patients, those cleared
   by the anchored policy and not by the hand-built policy, against the
   reverse. More cleared, with an exact two-sided McNemar p < 0.05.
2. **Dice non-inferiority.** Paired dDice: the lower 95% bound must be >= -0.0025.
3. **Lesion non-inferiority.** Lesion hits retained (anchored - hand-built),
   summed over tumour-positive cases, with a case bootstrap: the lower 95%
   bound must be >= -5.
4. **Beats the hand-built policy overall.** 1, 2 and 3 all hold.

**Validation** is a direction check, required for test:
- tumour-free cleared >= the hand-built policy's;
- paired dDice >= -0.0025;
- lesion hits retained >= -1.

**Test** is read **once**, and only if 1-3 hold out-of-fold and validation
passes. It is read for this configuration and for one hand-built test run
(`phase11_eag_hb_test.npz`, the route verified in Phase XI). **On test,
criteria 1-4 are the confirmatory result.** If anything fails first, test
stays untouched.

## Pre-specified descriptives (CAVE-PET)

- **Experience faithfulness** (paper 7). At the component level, held-out,
  compare the selected rule with an identical rule whose harm scores are
  shuffled among the eligible candidates in that fold (seed 2026, 20
  shuffles). If the learned experience matters causally, the shuffled rule
  should give lower summed `d_qdice` and lose more lesions.
- **Self-improvement curve.** At the component level, held-out: the harm
  model is refitted on 10%, 25%, 50% and 100% of the training patients
  (patient-sampled, seed 2026), with (`g`, `h`) fixed at that fold's
  selection. Report the added set's summed `d_qdice` and lesions lost at each
  size.
- **Calibration.** ECE (10 equal-mass bins) and Brier score of the harm model
  on the added candidates, held-out.
- **Subgroups.** The Phase XI proxy subgroup table, with the same flag.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
