# Phase IV - a diagnose-and-repair agent over the Phase IIIb ensemble

Branch `phase4-agent`, from `phase3b-cv`. Everything below is measured against
`reports/phase4_analysis_plan.md`, which was committed before any Phase IV number
was computed. Where a pre-registered criterion fires, it is reported as it fired,
including where that verdict is negative.

M0 throughout is the Phase IIIb 5-fold ensemble: test Dice 0.6638, FP 9.10 mL.

## Summary

Neither pre-registered hypothesis survives. **H3 is not supported** in any
configuration: the pre-registered agent accepts 0 actions in 738 out-of-fold cases
and is a bit-identical no-op, and every configuration that does act has a test
Dice interval straddling zero. **H4 is not supported** anywhere. **Q is falsified
as a judge of individual actions** at 37.1% precision against a pre-registered 60%
threshold, and per the plan the reflection gate is reported unusable rather than
retuned. The Dataset506 refinement network is **rejected on test**, losing
0.0360 Dice while roughly tripling false-positive volume.

Two things did come out of it.

**The loop reliably removes false positives.** Significant on all three splits in
every configuration that acts, clearing every false positive from 49 of 385
out-of-fold and 6 of 90 test tumour-free patients. The pre-registered primary
endpoint - Dice over tumour-**positive** cases - cannot see this by construction.

**The accept/reject decision was on the wrong signal.** Routing it through the
diagnostic head's own confidence instead of Q turns out-of-fold Dice from
-0.0045 [-0.0113, -0.0001] into **+0.0020 [+0.0009, +0.0033]** and 105 lost
lesions into **none**, while keeping the false-positive benefit. That change is
post-hoc and is reported as a separate configuration, not as the agent.

Sections 1-9 and 12 are pre-registered analysis. Sections 10, 13 and the
ban-lifted policy are post-hoc and labelled at every point of use.

---

## 1. What the model actually gets wrong (Step 1)

`scripts/phase4_diagnose.py` labels every predicted component and every ground-truth
lesion on the 738 out-of-fold scans against the seven failure classes the plan
defines, and prices each class by an **oracle**: the Dice the case would reach if
that class alone were repaired perfectly, leaving every other error in place.

Out-of-fold, over the 353 tumour-positive cases (mean Dice 0.6888):

| failure class | cases | items | mean Dice | oracle | headroom |
|---|---:|---:|---:|---:|---:|
| spurious_physio | 13 | 37 | 0.6655 | 0.6676 | +0.0021 |
| spurious_liver_spleen | 20 | 54 | 0.7119 | 0.7265 | +0.0146 |
| spurious_other | 284 | 2490 | 0.6765 | 0.7303 | +0.0538 |
| **missed** | 263 | 3347 | 0.6738 | 0.7997 | **+0.1259** |
| under_seg | 122 | 431 | 0.6937 | 0.7887 | +0.0950 |
| over_seg | 90 | 464 | 0.7565 | 0.8266 | +0.0701 |
| boundary | 205 | 573 | 0.7297 | 0.8242 | +0.0945 |
| fragmentation | 113 | 253 | 0.7500 | - | topology only |
| merging | 87 | 152 | 0.7577 | - | topology only |

Validation (38 positive, Dice 0.7061) agrees on the ordering: missed +0.1239,
under_seg +0.1339, boundary +0.0972, over_seg +0.0703, spurious_other +0.0108.

**Missed lesions dominate the headroom and nothing else is close.** Removing every
spurious component perfectly - all three classes, 2581 components - buys +0.0705.
Finding the missed lesions buys +0.1259 on its own.

Two bugs were found and fixed here, both detected because oracle Dice came out
above 1.0: a component touching several lesions was subtracted once per lesion, and
a component that was *not* removed but already overlapped a perfected lesion was
counted twice. The fix required storing per-component/per-lesion overlaps
(`gt_overlap`) rather than per-component totals. No oracle value now exceeds 1.0.
The small-component hypothesis for `PETCT_c2a51a65c6` was checked and disproved -
that case has no sub-5-voxel components at all.

## 2. Can the failures be spotted without the ground truth?

`scripts/phase4_diag_model.py` fits one classifier per failure class on 26 observable
features - no `z_rel`, no `u_epi`, nothing that needs the label. Patient-grouped CV
on out-of-fold, checked once on val:

| class | CV AUROC | val AUROC |
|---|---:|---:|
| spurious_physio | 0.9668 | 0.9992 |
| spurious_liver_spleen | 0.9646 | 0.9968 |
| merging | 0.8914 | 0.8837 |
| spurious_other | 0.8432 | 0.8805 |
| boundary | 0.8245 | 0.7870 |
| over_seg | 0.8051 | 0.8492 |
| under_seg | 0.8009 | 0.7731 |
| fragmentation | 0.7076 | 0.6925 |
| **missed** (recover head) | **0.6470** | **0.7305** |

**The central tension of Phase IV is in these two tables.** The classes that can be
detected almost perfectly - physiological and liver/spleen false positives, AUROC
0.96-0.99 - are worth +0.0021 and +0.0146. The class worth +0.1259 is the one the
model can barely detect, AUROC 0.647. An agent can only act where it can see, and
where it can see there is almost nothing to win.

## 3. The strategy library and how each strategy was measured (Step 2)

`scripts/phase4_strategies.py` holds nine CPU strategies - `topology_remove`,
`topology_fill`, `topology_merge`, `recover`, `boundary`, `suv41`, `suv_percist`,
`volume_grow`, `volume_shrink` - plus a tenth, `roi_repredict`, on the GPU.

`scripts/phase4_strategy_table.py` applies every applicable strategy to every
component on a fresh copy of the state and records the exact effect: **33,799
measured actions over 661 cases**. Dice, FP volume and lesion hits are updated from
the changed bounding box alone rather than rescored over the whole volume, with a
per-lesion touch count so a lesion counts as lost only when nothing else still
holds it.

Two correctness checks were built in rather than assumed.

**Feature agreement.** Delta-Q is only meaningful if the features recomputed here
match the ones Q was trained on. The script recomputes features for every
*unmodified* component and compares them against the candidate CSV. This caught a
real discrepancy: `dist_physio_mm` differed by 6.93 mm because `build_candidates.py`
takes `.min()` over the component and this code took `.mean()`. After the fix all
four probed features agree **exactly - mean absolute difference 0.0** over all 6,462
components.

**Two Dice conventions, kept apart.** On a tumour-free case, Dice is 0/0 the moment
a strategy empties the mask - which is precisely the outcome removal exists to
produce. Collapsing that to `nan` poisons the mean and hides removal's best use;
scoring it as 1.0 would misstate the primary endpoint. The table therefore carries
both: `d_dice`, which is `nan` on tumour-free cases because the pre-registered
primary endpoint is Dice over tumour-**positive** cases only, and `d_qdice`, which
scores an empty prediction on a tumour-free case as 1.0 - the target Q was actually
fitted against, and so the only fair basis for asking whether Q is gamed.

This mattered. The first policy run, before the split, reported *nothing helps* for
all three spurious classes. 64 `topology_remove` actions had returned `nan` - every
one a tumour-free case left clean - and a single `nan` in `np.mean` had deleted
removal's entire case for itself.

## 4. Q is not a usable judge of individual actions

The plan fixed this test in advance: Q is not a usable judge if Delta-Q > 0
coincides with Delta-Dice > 0 at below 60% on OOF, in which case the reflection gate
is reported as unusable rather than tuned until it works.

Measured against `d_qdice`, the target Q was trained on, over all 33,799 actions:

| strategy | n | dQ>0 | honest | gamed | precision |
|---|---:|---:|---:|---:|---:|
| recover | 1086 | 312 | 33 | 135 | 19.6% |
| suv_percist | 5718 | 1571 | 342 | 808 | 29.7% |
| suv41 | 6382 | 1613 | 333 | 765 | 30.3% |
| topology_merge | 1674 | 185 | 51 | 91 | 35.9% |
| topology_remove | 6462 | 1362 | 408 | 614 | 39.9% |
| volume_shrink | 5962 | 1343 | 423 | 501 | 45.8% |
| volume_grow | 6134 | 602 | 180 | 193 | 48.3% |
| boundary | 307 | 13 | 2 | 2 | 50.0% |
| topology_fill | 74 | 8 | 4 | 4 | 50.0% |
| roi_repredict | 1200 | 184 | 99 | 70 | 58.6% |
| **pooled** | **34999** | **7193** | **1875** | **3183** | **37.1%** |

**37.1% against a 60% threshold. Not one strategy clears the bar** - the closest, `roi_repredict` at 58.6%, misses it. Spearman(dQ,
dDice) over all actions is 0.0745.

The mechanism is worth stating because it is not a fitting failure. **The median
action moves Dice by 0.00021** - two parts in ten thousand. Q's own fit error
(Phase IIIb: R-squared 0.6305) is two to three orders of magnitude larger than the
quantity it is being asked to resolve. Q ranks whole masks well - within-case
Spearman 0.6966 over randomly drawn masks - and ranking a mask is simply not the
same job as adjudicating one small edit to it. Phase IIIb validated the first and
Phase IV assumed the second followed. It does not.

Per the plan the gate is reported unusable. It was not retuned.

## 5. Strategy 7: a refinement network (Dataset506)

The one strategy that is not a hand-written edit is a second network trained to
correct the first. `scripts/build_refine_dataset.py` builds
`Dataset506_AutoPET_Refine`: 814 cases, four input channels - PET, CT, the M0
probability map `p_bar`, and the stability uncertainty `u_stab` - with PET, CT and
labels hard-linked from Dataset505 (2442 links, 2 GB). Training used the MAE
pre-trained weights widened from 2 to 4 input channels by
`scripts/expand_input_channels.py`; the stem convolution turned out to appear under
**four** aliases in the checkpoint (`...convs.0.conv.weight`,
`...convs.0.all_modules.0.weight`, and both again under `decoder.encoder.`), and all
four had to be widened with zeros or nnU-Net rejected the load. Weight norm is
unchanged at 8.619869.

Channels 2 and 3 for the 76 validation cases come from *single-model* maps rather
than the ensemble, deliberately: Phase IIIb measured ensemble `u_stab` at roughly
half its single-model value, so the training set would otherwise mix two
distributions. This is recorded in the dataset description.

500 epochs, 8 h 21 min, 59.7 s/epoch. Applied with `checkpoint_final.pth` - not
`checkpoint_best` - so that nothing is selected on the validation cases.

Because the network is one model and M0 is five, and Phase IIIb priced that gap at
+0.0213 Dice on its own, it is applied twice: once on ensemble inputs (what a
deployed system would have) and once on single-model inputs (distribution-matched,
one model against one model). Both are reported, so that "the refinement network
does not work" is separable from "the refinement network was fed a distribution it
never saw".

Validation, paired bootstrap over patients, 2000 resamples. Dice and HD95 over the
38 tumour-positive cases; FP and FN volume over all 76, per the plan:

| variant | dDice | dFP mL | dFN mL | dHD95 | d lesion sens |
|---|---|---|---|---|---|
| ensemble inputs | **+0.0131 [-0.0081,+0.0387]** | +10.64 [+4.16,+18.86] | -1.11 [-2.45,-0.15] | +17.64 [-8.72,+52.68] | +0.0028 |
| single inputs | **+0.0092 [-0.0020,+0.0235]** | +4.92 [+2.35,+8.02] | -0.63 [-1.82,+0.09] | +3.19 [-0.04,+8.77] | -0.0138 |

**Both Dice intervals straddle zero, so by the plan's own wording this is "no effect
demonstrated", not a trend.** Every interval that does exclude zero is a cost: false
positive volume rises sharply on both variants, and tumour-free patients carrying at
least one false positive go from 22/38 to 30/38 on ensemble inputs. The network is
more aggressive - it finds more tumour volume (FN down) and pays for it with far
more spurious volume.

The single-input variant additionally **fails the pre-registered safety constraint**:
lesion sensitivity -0.0138 against a -0.01 margin. It is ineligible regardless of
its Dice.

On the case split, ensemble inputs improve 17, leave 9 unchanged and degrade 12 of
38 - a wide two-sided spread rather than a consistent gain, which is what an
interval straddling zero looks like at the case level.

### On test, it is not ambiguous

The refinement network was carried into the single test pass because its ensemble
variant met the safety margin on validation. Over 200 test cases, paired bootstrap
over patients:

| | ensemble inputs | single inputs |
|---|---|---|
| Dice | **-0.0360 [-0.0635, -0.0132]** | **-0.0327 [-0.0584, -0.0108]** |
| FP volume | **+20.06 mL [+13.39, +27.14]** | **+16.40 mL [+10.87, +22.80]** |
| FN volume | -0.82 mL [-1.69, -0.20] | -0.35 [-1.15, +0.40] |
| lesion sensitivity | **+0.0384 [+0.0160, +0.0576]** | -0.0022 [-0.0114, +0.0107] |
| FP lesions per patient | +2.32 [+1.90, +2.80] | +1.35 [+1.06, +1.65] |
| tumour-free with any FP | **54/90 -> 72/90** | 54/90 -> 67/90 |
| improved / unchanged / degraded | 26 / 26 / 58 | 30 / 31 / 49 |

**What validation left ambiguous, test settles: the refinement network is harmful.**
Both variants lose Dice with intervals excluding zero, and both roughly triple
false-positive volume. On ensemble inputs it adds 18 tumour-free patients to the
set carrying at least one false positive.

The one real gain is lesion sensitivity, **+0.0384 [+0.0160, +0.0576]** on ensemble
inputs - the network genuinely finds lesions M0 misses, which is the hardest
problem in section 1. It just brings 2.32 extra false-positive lesions per patient
with them. At that exchange rate the finding is a direction for future work, not a
usable component.

Validation had called this "no effect demonstrated" with FP already rising. Test
confirmed the cost and added a Dice loss that validation's 38 tumour-positive
cases could not resolve. Nothing here was selected on test; the network was frozen
at `checkpoint_final.pth` before validation was scored.

## 6. Strategy 6: re-predicting a single region of interest

`roi_repredict` is the other learned strategy, and the only one that puts the
*original* network back to work: it crops a box around one component, re-runs M0's
fold-0 weights on that crop alone, and replaces the component with the result. The
point is that a crop is a different input from a whole body - the patch sampler
sees the lesion at a different position and the normalisation statistics are local.

Getting this right took a fix that is worth recording, because the broken version
did not crash. The first implementation handed **raw** SimpleITK arrays to
`predict_logits_from_preprocessed_data` and cropped a 24 mm margin around the
component. The output looked like a probability map and was silently almost zero:
`P.max() = 3.3e-8` where the stored `p_bar` for the same component was 0.997. Two
diagnostics separated the causes - a full-volume control reproduced `p_bar`
exactly (201 voxels above 0.5 in both), which ruled out the weights and the
preprocessing, leaving the crop. A 24 mm margin yields boxes far smaller than the
192x96x128 patch the network was trained with, and nnU-Net pads the remainder with
background. The fix routes the crop through `DefaultPreprocessor.run_case` (taking
the offsets from `bbox_used_for_cropping` and asserting the extent) and expands
every box to at least one full patch. After the fix a 12-component smoke test gave
10 improved, 0 worsened, 0 lesions lost.

Out-of-fold: **1,200 components over 255 cases, 9.6 s per component**, 1,156 of them
on tumour-positive cases.

| failure class | n | mean dDice | median dDice | % positive |
|---|---:|---:|---:|---:|
| under_seg | 264 | **+0.0088** | +0.00025 | 78.0 |
| fragmentation | 305 | **+0.0059** | +0.00032 | 72.1 |
| boundary | 499 | +0.0014 | +0.00007 | 60.5 |
| none | 188 | +0.0005 | -0.00000 | 47.9 |
| over_seg | 146 | -0.0010 | -0.00003 | 39.7 |
| merging | 116 | -0.0011 | -0.00008 | 44.8 |
| spurious_liver_spleen | 4 | -0.0013 | -0.00039 | 0.0 |
| spurious_other | 56 | **-0.0034** | -0.00097 | **5.4** |

This is the one strategy whose behaviour matches the diagnosis instead of
contradicting it. It helps exactly the classes that are about *shape* - under
segmentation and fragmentation, where the network already found the lesion and
drew it too small or in pieces - and it hurts the spurious classes, where
re-running the same network on a crop simply re-draws the same false positive
(5.4% positive on `spurious_other` is the cleanest negative result in the table).
Net over all components: +0.0026 mean dDice, **5 lesions lost against 22 gained**,
at a cost of +0.217 mL false positive volume per component.

Two things keep this from being the answer.

**It is still tiny.** 775 of 1,156 components move Dice by less than 0.001. The
mean is carried by 65 components above +0.01 against 26 below -0.01.

**The agent as run could not execute it.** `roi_repredict` is not in the agent's
`REGISTRY` - it was added to the strategy table after the agent runs were already
launched, and at 9.6 s per component a full agent run over 738 cases with three
actions each would have cost roughly six GPU-hours on top. It is therefore reported
here as a **per-component counterfactual study, not a case-level result**: each row
is the effect of changing that one component in isolation, and composing three of
them on one case is not measured. That caveat is load-bearing and the numbers above
should not be read as a case-level Dice gain.

It does, however, change what the pre-registered policy would have contained. With
`roi_repredict` in the table it survives the gaming ban (99 honest against 70 gamed)
and ranks **first for `under_seg`, `fragmentation` and `boundary`** - the three
largest ground-truth-overlapping classes. The pre-registered policy without it
(below) has nothing to offer those classes at all.

## 7. The policy, and the cost of the pre-registered gaming ban

Section 9 of the plan fixed a gaming check in advance: *remove any strategy whose
gaming rate exceeds its honest rate*. Applied to the measured table that rule bans
**seven of the ten strategies** - `recover`, `suv41`, `suv_percist`,
`topology_merge`, `topology_remove`, `volume_grow`, `volume_shrink` - leaving
`boundary`, `topology_fill` and (had it been available) `roi_repredict`.

The two policies that follow are reported side by side. The **pre-registered**
policy is primary. The **ban-lifted** policy is explicitly post-hoc, and exists to
answer one question the pre-registered policy cannot: whether the agent's failure
is a failure of the loop or an artefact of the ban.

| failure class | pre-registered policy | ban-lifted policy (post-hoc) |
|---|---|---|
| spurious_physio | *(nothing helps)* | suv_percist, suv41, topology_remove, volume_shrink |
| spurious_liver_spleen | *(nothing helps)* | topology_remove, suv_percist, suv41, volume_shrink |
| spurious_other | *(nothing helps)* | topology_remove, suv_percist, volume_shrink |
| missed | *(nothing helps)* | recover |
| under_seg | boundary | volume_grow, topology_merge, suv41, boundary |
| over_seg | topology_fill | topology_fill |
| fragmentation | *(nothing helps)* | volume_grow, topology_merge, boundary |
| merging | topology_fill | topology_fill |
| boundary | boundary, topology_fill | volume_grow, boundary, topology_fill |

The pre-registered policy has an answer for three of nine failure classes, and the
two strategies it keeps are the two that were measured on the fewest actions in the
whole library (`boundary` n=307, `topology_fill` n=74). Both reach exactly 50%
precision on 4 and 8 informative actions respectively - they are not kept because
they are good, they are kept because too little was measured about them to convict
them. **The gaming ban selected for strategies with small sample sizes.** That is a
real defect in the pre-registered rule, and it is reported rather than repaired.

The ban is also not the only guard. The safety constraint independently drops
`topology_remove` from every ground-truth-overlapping class, which is where its two
largest honest gains sit (+0.0113 dDice on `spurious_liver_spleen`, +0.0088 on
`spurious_other`). Those gains are on *spurious* classes, so the safety guard is
behaving correctly - but the two rules together leave very little standing.

Figure: `fig_strategy_heatmap_s6.png`, the mean dDice of every strategy on every
failure class, with `roi_repredict` included;
`fig_strategy_heatmap.png` is the same without it.

(The heatmap is identical under both policies, because it plots what was
*measured*, and the ban changes only which of those measurements the policy is
allowed to use. `fig_strategy_heatmap_noban.png` and
`fig_strategy_heatmap_prereg_s6.png` are byte-identical duplicates of the two
above - the policy script writes one per run - and carry no extra information.)

## 8. The agent on out-of-fold and validation (Step 4)

Two configurations were run, each over all 738 out-of-fold and 76 validation cases,
T=3 actions per case, epsilon=0.005.

**A - pre-registered**: pre-registered policy, reflection gate on.
**B - post-hoc**: ban-lifted policy, gate off, with tau_Q applied afterwards from
the recorded `q0` so that one permissive run yields the whole threshold sweep.

### A. The pre-registered agent is a literal no-op

Over 738 out-of-fold cases the pre-registered agent **attempted 5 actions and
accepted none**. Over 76 validation cases it attempted none. Every number it
produces is M0's number, to the last digit:

| | Dice | FP mL | FN mL | HD95 | lesion sens | FP/patient | tumour-free with FP |
|---|---|---|---|---|---|---|---|
| M0 (OOF) | 0.6888 | 13.256 | 6.589 | 111.29 | 0.5454 | 3.50 | 289/385 |
| agent A (OOF) | 0.6888 | 13.256 | 6.589 | 111.29 | 0.5454 | 3.50 | 289/385 |

dDice +0.0000 [+0.0000, +0.0000]. The interval does not exclude zero, so **H3 is not
supported by the pre-registered configuration** - not because the agent made things
worse, but because it never acted. Every state of the iteration ladder is identical
for the same reason.

The cause is section 7: the policy had something to offer only `under_seg`,
`over_seg`, `merging` and `boundary`, those are a small minority of the proposals,
and the 2-of-3 acceptance rule plus the epsilon threshold removed the rest.

### B. The ban-lifted agent acts, and loses Dice

Lifting the ban gives the agent 1,373 accepted actions over 659 of 738 cases. The
tau_Q sweep (tuned on out-of-fold, checked on val, never on test):

| tau_Q | cases acted | OOF Dice | OOF dSens | OOF FP mL | val dSens | eligible |
|---|---:|---:|---:|---:|---:|---|
| 0.00 | 84 | **0.6843** | -0.0004 | 12.918 | +0.0000 | yes |
| 0.30 | 322 | 0.6761 | -0.0021 | 10.544 | -0.0055 | yes |
| 0.50 | 381 | 0.6748 | -0.0029 | 9.775 | -0.0069 | yes |
| 0.70 | 506 | 0.6749 | -0.0047 | 9.450 | -0.0083 | yes |
| 1.00 | 736 | 0.6752 | -0.0047 | 9.238 | -0.0083 | yes |

Every threshold passes the -0.01 safety margin on both splits, so the rule selects
the best out-of-fold Dice: **tau_Q = 0.00**. Note what that threshold means. `q0` is
M0's predicted Dice for the case, so tau_Q = 0 lets the agent act only on cases Q
believes are already worthless - 84 of 738. The sweep's own answer to "when should
this agent act?" is "almost never".

At tau_Q = 0.00, paired bootstrap over 738 patients, 2000 resamples:

| | agent B - M0 (OOF) |
|---|---|
| Dice | **-0.0045 [-0.0113, -0.0001]** |
| FP volume | **-0.3375 mL [-0.5246, -0.1797]** |
| FN volume | +0.0785 mL [+0.0000, +0.2301] (straddles 0) |
| lesion sensitivity | -0.0004 [-0.0011, +0.0000] (straddles 0) |
| FP lesions per patient | **-0.1274 [-0.1680, -0.0908]** |

**The Dice interval excludes zero in the wrong direction.** The plan's criterion for
H3 was an interval excluding zero; this one does, and the effect is harm. H3 is
rejected on the post-hoc configuration too, more informatively than on the
pre-registered one.

The ladder tells the same story - *every* rung sits below M0, and no rung beats
another by a margin worth having:

| state | OOF Dice | OOF FP mL |
|---|---|---|
| M0 (no action) | 0.6888 | 13.256 |
| after 1 action | 0.6856 | 13.151 |
| after 2 actions | 0.6885 | 13.163 |
| after 3 actions | 0.6871 | 13.172 |
| agent (best-Q state kept, tau_Q=0) | 0.6843 | 12.918 |

**This is an iteration-budget ladder, not the section 11 ablation ladder, and the
two should not be confused.** `M1`/`M2`/`M3` in the evaluation output are the
state of the mask after the first, second and third accepted action - they are not
"+uncertainty", "+reasoning" and "+refinement". Spending more of the budget does
not help and the curve is not monotone, which is itself informative: the first
action is on average the worst one, and the agent's best-Q selection rule does not
recover from it.

The section 11 ladder as pre-registered was only **partially** executed. M0 and
the two gate rungs exist as real configurations - "+refinement" is the loop with
every proposed action accepted (configuration B, gate off) and "+reflection" is
the same loop with the Q gate on (configuration A, and the tau_Q sweep). The
"+uncertainty" and "+reasoning" rungs were never run as separate configurations:
isolating them means retraining the diagnosis model without its uncertainty
features, which was not done. Those two rungs are **not reported**, and the
pre-registered criterion that compares them cannot be evaluated from this run.

`fig_ablation_ladder.png` plots both columns. Note the Dice axis spans 0.0045 in
total - the visual drop is the whole effect, and the whole effect is small.

### What the agent does well

One secondary endpoint is unambiguously positive, on both splits, with intervals
excluding zero: **the agent removes false positives.**

| | OOF | val |
|---|---|---|
| FP volume | 13.256 -> 12.918 mL | 5.158 -> 4.954 mL |
| FP lesions per patient | 3.50 -> 3.37 | 2.05 -> 2.01 |
| tumour-free patients with any FP | **289/385 -> 227/385** | **22/38 -> 16/38** |

62 of 385 tumour-free out-of-fold patients, and 6 of 38 on validation, go from
carrying at least one false positive to carrying none. That is the single clinically
legible thing the loop achieves, and `topology_remove` on `spurious_other` does
almost all of it (1,003 of 1,373 actions).

The outcome split explains why this does not show up in Dice: on tumour-positive
cases the agent **improves 0, leaves 349 unchanged and degrades 4** (|dDice| < 0.01
counts as unchanged). It barely touches the cases Dice is measured on. The
false-positive work happens almost entirely on tumour-free patients, which the
primary endpoint by construction excludes.

### Per-action effects

| action | n | mean dDice | lesions lost |
|---|---:|---:|---:|
| topology_remove | 1036 | -0.01144 | 105 |
| recover | 149 | -0.00136 | -14 (recovered) |
| volume_grow | 109 | **+0.00470** | 0 |
| boundary | 41 | -0.00026 | 0 |
| suv_percist | 19 | +0.00035 | 0 |
| suv41 | 12 | -0.00488 | 0 |
| topology_merge | 6 | +0.00201 | 0 |
| topology_fill | 1 | +0.00056 | 0 |

`topology_remove` is doing the useful false-positive work and paying for it with
105 lost lesions and the entire Dice deficit. `volume_grow` is the only action with
a clearly positive mean, and it fires 109 times.

By the size of the component removed:

| volume removed | n | mean dDice | lesions lost | total mL removed |
|---|---:|---:|---:|---:|
| < 0.5 mL | 624 | -0.00009 | 42 | 133 |
| 0.5 - 2 mL | 264 | **-0.03312** | 43 | 268 |
| 2 - 10 mL | 94 | -0.02257 | 16 | 420 |
| 10 - 50 mL | 40 | -0.01305 | 2 | 845 |
| > 50 mL | 14 | **-0.05963** | 2 | 1819 |

The two ends behave completely differently. 624 removals of sub-0.5 mL specks
clear 133 mL between them and cost almost nothing on Dice, but still drop 42
lesions - these are small true lesions being deleted one at a time. The 14
removals above 50 mL clear 1,819 mL, more than all other bins combined, and carry
the worst mean of any bin because when one of them is wrong it is wrong on a
case's entire tumour burden (`PETCT_0e2034240b` in section 10 is one of these 14).
The middle band, 0.5 to 2 mL, is the worst value per millilitre: 264 actions, 268
mL cleared, 43 lesions lost.

`recover` ran 149 times and recovered 14 lesions, at a mean of -0.00136 Dice. Its
71 sub-0.2 mL recoveries are free but recover only 7 lesions; the 17 between 0.2
and 1 mL recover 6 lesions and cost -0.01877 each.

## 9. Where the damage lands (section 22, H4)

H4 asked whether the agent helps most on the cases the base model finds hardest.
Cases were split into Q tertiles using boundaries fixed out-of-fold (q0 = 0.130 and
0.680) and applied unchanged everywhere.

| tertile | n | dDice (agent B - M0, OOF) |
|---|---:|---|
| low Q (hardest) | 246 | **-0.0544 [-0.1328, -0.0006]** |
| mid Q | 246 | +0.0000 [+0.0000, +0.0000] |
| high Q (easiest) | 246 | +0.0000 [+0.0000, +0.0000] |

**H4 is rejected, and the mechanism is the selection rule itself.** Because the
sweep chose tau_Q = 0, the agent acts only where Q predicts a near-zero Dice - which
is the low tertile, by definition. The mid and high tertiles are untouched to the
last digit. So the whole of the agent's effect is concentrated on the hardest cases,
and on those cases it costs **0.0544 Dice** with an interval excluding zero.

This is worse than the pooled -0.0045 suggests. The pooled figure is diluted by the
two thirds of cases the agent never touches. Conditional on acting, the agent is
substantially harmful. H4 predicted the opposite sign.

The same argument read forwards explains why tau_Q = 0 won the sweep at all: the
out-of-fold Dice criterion prefers the threshold that *acts least*, and tau_Q = 0 is
the least-acting setting on the grid. The sweep was not finding a good operating
point; it was finding the closest available approximation to doing nothing.

## 10. The gate was on the wrong signal (post-hoc)

Two out-of-fold cases, with nearly identical observable features and opposite
outcomes, say more than the aggregate numbers do. Both are the agent's first
action; both are `topology_remove` on a large component the diagnostic head
labelled `spurious_other`.

```
PETCT_0e2034240b                  7 GT lesions
  it1  topology_remove  spurious_other  p_fail=0.579  vol=54.17 mL  SUVmax=13.13
       dQ=-0.2859   dDice=-0.53288   lesions lost 1   ACCEPTED
  -> Dice 0.5329 -> 0.0000 to clear 0.26 mL of false positive

PETCT_fde79b6aa9_20030816_53988178    tumour-free
  it1  topology_remove  spurious_other  p_fail=0.967  vol=187.85 mL  SUVmax=9.99
       dQ=-0.2029   dDice=+0.00000   lesions lost 0   ACCEPTED
  -> FP 188.28 mL -> 0.00 mL, nothing true removed
```

The first action is the worst in the entire run; the second is close to the best.
**Q scored them almost the same, and scored both negative** (-0.286 against -0.203).
A critic that assigns its second-most-negative score to the single most valuable
action available is not mis-tuned, it is measuring something else - which is exactly
what section 4 found at scale, here in one pair of cases.

The feature that *does* separate them is the diagnostic head's own confidence:
**0.579 against 0.967**. Stratifying all 1,036 out-of-fold `topology_remove`
actions by it:

| p_failure | n | mean dDice | lesions lost |
|---|---:|---:|---:|
| 0.00-0.60 | 194 | **-0.08523** | **70** |
| 0.60-0.80 | 280 | +0.00428 | 30 |
| 0.80-0.90 | 142 | +0.00212 | 4 |
| 0.90-0.95 | 74 | +0.00476 | 1 |
| 0.95-1.00 | 346 | +0.00318 | 0 |

**The entire Dice deficit lives in the lowest bin.** 194 of 1,036 actions - 19% -
carry a mean of -0.085 and two thirds of all 105 lost lesions. Every bin above 0.6
is positive. As a threshold:

| guard | actions kept | mean dDice | lesions lost |
|---|---:|---:|---:|
| none | 1036 (100%) | -0.01144 | 105 |
| p_failure >= 0.60 | 842 (81%) | +0.00349 | 35 |
| p_failure >= 0.80 | 562 (54%) | +0.00312 | 5 |
| **p_failure >= 0.90** | **420 (41%)** | **+0.00346** | **1** |
| p_failure >= 0.95 | 346 (33%) | +0.00318 | 0 |

The direction replicates on validation - 76 `topology_remove` actions, mean dDice
-0.00540 with 11 lesions lost unguarded, against +0.00144 with 1 lesion lost at
p_failure >= 0.90.

This is **post-hoc and must be read as such**: the threshold was chosen by looking
at out-of-fold outcomes, it was not pre-registered, and the per-action deltas above
are measured against sequential state, so dropping actions is not strictly additive.
The honest way to price it is to re-run the loop with the guard in place, which is
section 11.

`fig_pfail_guard.png` plots both columns of the stratification.

Section 2 found the diagnostic head well calibrated on the spurious classes
(AUROC 0.96-0.97) and Q poorly suited to adjudicating single edits. The architecture
nonetheless routed the accept/reject decision through Q and used `p_failure` only to
*order* proposals. On this evidence that is the wrong way round.

## 11. Structural limits of the design, independent of the results

Three of these were visible before any number came in and should have been priced
at design time.

**The budget cannot reach the errors.** The loop takes at most T=3 actions per
case. Out-of-fold there are 2,490 `spurious_other` items over 284 cases (8.8 per
case) and 3,347 `missed` items over 263 cases (12.7 per case); the median case
carries 3 predicted components and 7 ground-truth lesions, the 90th percentile 19
components. Three actions against roughly twenty items caps the achievable gain
far below the oracle headroom whatever the policy does. The agent was never
given enough moves to win.

**The headroom and the detectability are in different places.** Section 1 put the
largest oracle gain on `missed` (+0.1259) and section 2 put its AUROC at 0.647,
the worst of the nine classes. The classes that can be spotted - `spurious_physio`
0.967, `spurious_liver_spleen` 0.965 - are worth +0.0021 and +0.0146. The one
strategy addressing `missed`, `recover`, has the worst Q precision in the library
(19.6%) and a mean effect of -0.00136 over 149 accepted actions. This is the
central tension of the whole phase and no amount of policy tuning dissolves it.

**The primary endpoint cannot see the agent's one real success.** Dice over
tumour-positive cases is, by construction, blind to a tumour-free patient whose
188 mL of false positive is removed. 62 of 385 tumour-free out-of-fold patients are
cleaned entirely, and the primary endpoint scores that as nothing. The endpoint is
not wrong - it was pre-registered for good reasons - but it is the wrong instrument
for the thing the loop turned out to be good at, and that mismatch was foreseeable
once `topology_remove` became the dominant action.

Two further limits are specific to what was run.

**Strategy 6 is a per-component counterfactual, not a case-level result**, and it
uses M0's fold-0 weights alone rather than the five-member ensemble. Its +0.0026
mean should not be added to anything.

**tau_Q = 0 and p_failure >= 0.90 were both chosen by looking at out-of-fold
outcomes.** tau_Q was pre-registered as a tuned quantity, so that is within the
plan; the p_failure floor was not, and section 12 reports it as the post-hoc
variant it is.

## 12. The single test pass

Test was touched once, after everything above was fixed. Nothing below was
selected on test: the policy, tau_Q = 0.00 and the p_failure floor were all fixed
on out-of-fold data and checked on validation first. 200 patients, 110 of them
tumour-positive, 90 tumour-free.

### Configuration A - pre-registered

Over 200 test cases the pre-registered agent attempted 2 actions and neither moved
a metric at four decimal places. Every rung of the ladder is M0:

| | Dice | FP mL | FN mL | HD95 | lesion sens | tumour-free with FP |
|---|---|---|---|---|---|---|
| M0 | 0.6638 | 9.084 | 5.270 | 120.69 | 0.5495 | 54/90 |
| agent A | 0.6638 | 9.084 | 5.270 | 120.69 | 0.5495 | 54/90 |

dDice +0.0000 [+0.0000, +0.0000]. **H3 is not supported.**

### Configuration B - ban-lifted, tau_Q = 0.00 (post-hoc policy)

404 accepted actions over 21 of 200 cases.

| | agent B - M0 (test) |
|---|---|
| Dice | +0.0005 [-0.0055, +0.0075] (straddles 0) |
| FP volume | **-0.8963 mL [-1.8154, -0.2599]** |
| FN volume | +0.0004 mL [+0.0000, +0.0013] (straddles 0) |
| lesion sensitivity | -0.0006 [-0.0021, +0.0000] (straddles 0) |
| FP lesions per patient | **-0.0750 [-0.1500, -0.0100]** |
| tumour-free with any FP | **54/90 -> 42/90** |

improved 1, unchanged 108, degraded 1 of 110 tumour-positive cases. Safety is met
with room to spare (-0.0006 against -0.01). H4 is not supported on test either
(low-Q +0.0040 [-0.0439, +0.0528]).

**H3 is not supported on test by either configuration.** The pre-registered agent
cannot support it because it does not act; the ban-lifted agent does act, and its
Dice interval straddles zero.

One honest observation in the agent's favour: the out-of-fold harm did **not**
replicate. Out-of-fold the same configuration gave -0.0045 [-0.0113, -0.0001] and
on the hardest tertile -0.0544; on test the point estimate is +0.0005 and the
hardest tertile is +0.0040. Part of the out-of-fold deficit was the threshold
selection fitting its own noise - which is what an honest out-of-fold-to-test
comparison is for. The conclusion is unchanged, but it should be stated as *no
demonstrated effect on Dice*, not as *demonstrated harm*.

What survives the test pass is the secondary endpoint that survived everywhere
else: **the agent removes false positives**, with intervals excluding zero on all
three splits, and clears every false positive from 12 of 90 tumour-free test
patients.

## 13. The guarded loop (post-hoc), on all three splits

Section 10 argued the accept/reject decision was routed through the wrong signal.
`--min-pfail` tests that directly: the loop is re-run, unchanged except that a
proposal is only considered when the diagnostic head gives it at least 0.90. The
threshold was read off out-of-fold and checked on validation before test was
touched, but it was **not pre-registered**, so everything here is post-hoc and is
reported as a separate configuration rather than as the agent.

This is a re-run of the loop, not a filtered replay of the old one: dropping an
action changes what the next iteration sees, so the sequential state is rebuilt.

| | out-of-fold (738) | validation (76) | test (200) |
|---|---|---|---|
| Dice | **+0.0020 [+0.0009, +0.0033]** | +0.0005 [-0.0013, +0.0028] | +0.0046 [-0.0008, +0.0147] |
| FP volume mL | **-1.323 [-2.308, -0.582]** | **-0.169 [-0.416, -0.032]** | **-0.874 [-2.208, -0.086]** |
| FP lesions / patient | **-0.564 [-0.621, -0.505]** | **-0.290 [-0.421, -0.171]** | **-0.355 [-0.435, -0.275]** |
| lesion sensitivity | **+0.0000 (no lesion lost)** | -0.0014 | -0.0011 |
| tumour-free with any FP | **289/385 -> 240/385** | 22/38 -> 20/38 | **54/90 -> 48/90** |
| improved / unchanged / degraded | 17 / 334 / 2 | 1 / 36 / 1 | 4 / 105 / 1 |
| actions | 453 | 23 | 77 |

Three things changed at once, and all three in the same direction.

**The Dice loss became a Dice gain.** Out-of-fold the interval excludes zero on the
positive side: +0.0020 [+0.0009, +0.0033], against -0.0045 [-0.0113, -0.0001]
unguarded. On val and test the intervals straddle zero, so the gain does not
transfer as a demonstrated effect, but the sign is positive on all three splits
and the degraded count collapses - out-of-fold, 17 improved against 2 degraded,
where the unguarded loop managed 0 improved against 4.

**No lesion was lost at all out-of-fold.** Delta lesion sensitivity is exactly
+0.0000 across 738 cases and 453 actions, against 105 lost lesions unguarded. The
pre-registered safety margin, which the unguarded loop at full permissiveness
actually violates on test (-0.0106 against a -0.01 budget), stops being a
constraint that binds.

**The false-positive benefit survived.** It is significant on all three splits and
is now bought without the Dice cost: 49 more tumour-free out-of-fold patients
cleared than before, and 6 of 90 on test.

The tau_Q sweep makes the point a second way. Unguarded, the rule chose
**tau_Q = 0.00** - act almost nowhere. Guarded, it chooses **tau_Q = 1.00** - act
on 736 of 738 cases. Once proposals are filtered by the detector, the critic has
nothing left to veto, and the sweep stops trying to switch the agent off.

H4 is still not supported, but it fails differently: the hardest tertile now has
the *largest positive* point estimate on both out-of-fold (+0.0069) and test
(+0.0345), with intervals that include zero. The direction H4 predicted is there;
the evidence is not.

Note the honest ceiling. +0.0020 Dice out-of-fold is roughly a tenth of what
simply ensembling five models bought in Phase IIIb (+0.0213), and the test
interval includes zero. This is a loop that has stopped doing harm and does a
measurable amount of false-positive cleanup - not a loop that improves
segmentation.

## 14. Verdicts against the pre-registered plan

**H3 - the agent improves Dice over M0. NOT SUPPORTED.**

The plan required the paired bootstrap interval to exclude zero. On test:

| configuration | test dDice | verdict |
|---|---|---|
| A, pre-registered | +0.0000 [+0.0000, +0.0000] | not supported - the agent never acts |
| B, ban-lifted, tau_Q=0 | +0.0005 [-0.0055, +0.0075] | not supported - straddles zero |
| B at full permissiveness | -0.0225 [-0.0486, -0.0011] | excludes zero, **the wrong way** |
| guarded, p_failure >= 0.90 (post-hoc) | +0.0046 [-0.0008, +0.0147] | not supported - straddles zero |

No configuration supports H3. The closest is the post-hoc guarded one, which is
positive on all three splits and significant on out-of-fold only.

**H4 - the agent helps the hardest cases most. NOT SUPPORTED** on any split in any
configuration. Unguarded it fails with the wrong sign (-0.0544 on the hardest
out-of-fold tertile); guarded it fails with the right sign and intervals including
zero.

**Q as a judge of single actions. FALSIFIED**, exactly as the plan's section 12
anticipated: 37.1% of actions with dQ > 0 improve Dice, against a 60% threshold,
and no strategy clears the bar. Per the plan the gate is **reported unusable
rather than retuned**, and that is what was done.

**"The loop adds nothing if M1 and M3 are within 0.005 Dice." CANNOT BE
EVALUATED.** As section 8 records, the +uncertainty and +reasoning rungs were never
run as separate configurations, so the pre-registered ladder does not exist for
this run. The iteration-budget ladder that does exist shows more actions do not
help, which is a related but different claim.

**Deployability.** The plan said the agent is not deployable if no configuration is
eligible. Judged on the primary endpoint, **nothing here is deployable as a Dice
improvement**. Judged on the secondary endpoint it is a different answer: the
guarded configuration reduces false-positive volume and false-positive lesions per
patient with intervals excluding zero on all three splits, loses no lesions
out-of-fold, and clears every false positive from 49 out-of-fold and 6 test
tumour-free patients. If the deployment question is "fewer spurious findings at no
measurable cost", that configuration answers it. If it is "better segmentation",
nothing here does.

**The refinement network is rejected.** Dice -0.0360 [-0.0635, -0.0132] on test,
with false-positive volume roughly tripled. Its one real finding - lesion
sensitivity +0.0384 [+0.0160, +0.0576] - is a direction for future work at an
exchange rate no one would accept today.

### What this phase actually establishes

1. **Q ranks masks; it does not adjudicate edits.** Phase IIIb validated the first
   and Phase IV assumed the second followed. The median action moves Dice by
   0.00021, two to three orders of magnitude below Q's own fit error. The two
   example traces in section 10 are the compact proof: Q scored the single most
   valuable action in the run *more negatively* than the second-worst.

2. **The detector's own confidence is the usable gate.** Moving the accept/reject
   decision from Q to `p_failure` turns -0.0045 into +0.0020 out-of-fold and 105
   lost lesions into none, while keeping the false-positive benefit.

3. **The pre-registered gaming ban selected for small samples.** It kept the two
   strategies measured on the fewest actions (`boundary` n=307, `topology_fill`
   n=74, both at exactly 50% on 4 and 8 informative actions) and banned the seven
   measured on thousands. That is a defect in the rule, reported rather than
   repaired.

4. **Detectability and headroom sit in different places.** The biggest oracle gain
   is `missed` (+0.1259) at the worst AUROC (0.647); the best-detected classes
   (0.96-0.97) are worth +0.0021 and +0.0146. No policy dissolves that.

5. **Re-predicting a cropped region is the most promising untested idea.** It is
   the only strategy whose class profile matches the diagnosis - +0.0088 on
   `under_seg`, +0.0059 on `fragmentation`, -0.0034 on `spurious_other` - and the
   only one whose Q precision approaches the threshold (58.6%). It has never been
   run inside the loop.

### What Phase V should change

- **Gate on `p_failure`, not on Q**, and pre-register that.
- **Put `roi_repredict` in the REGISTRY** and measure it at case level rather than
  per component.
- **Raise T**, or drop the per-case budget entirely for removal actions. Three
  actions against 8.8 spurious and 12.7 missed items per case cannot win.
- **Report a false-positive endpoint as primary alongside Dice.** Dice over
  tumour-positive cases cannot see the one thing this loop does well.
- **Do not re-use a case-level quality model as an action-level critic** without
  measuring it at the scale of a single action first.
