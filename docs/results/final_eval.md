# Final evaluation - work plan sections 21 to 24


> **Note, 2026-10-24 - partly superseded by `audit_reanalysis.md`.**
> The out-of-fold columns of sections 1, 2 and 4 were computed with detectors
> fitted on all 738 out-of-fold scans and then applied to those same scans, so
> they are in-sample. Cross-fitted on the Phase IIIb folds 3-7:
>
> - **Section 4, "removing it is the right action 98% of the time"**: the figure
>   is **85.3%**, and lesions lost at the 0.90 gate rise from 3 to **41**. As a
>   per-action rule the honest threshold is >= 0.95; inside the loop, with the Q
>   gate filtering marginal actions, 0.90 still wins at case level.
> - **Section 1, + reasoning**: 0.6919 becomes 0.6876, **-0.0012
>   [-0.0064, +0.0025]**. Its claim to match matched-rate uncertainty on Dice does
>   not survive; its false-positive advantage does and is larger (40 patients
>   cleared against 15).
> - **Section 1, FINAL AGENT**: 0.6943 becomes **0.6938**, 240 becomes **239**.
>   Essentially unchanged.
> - **Section 2, H4**: survives at 95%, **fails Bonferroni** - both hardest-group
>   lower bounds +0.0000. Read as directionally supported, magnitude not
>   established.
> - **Section 4, self-improvement**: the conclusion holds on a common held-out
>   set with 20 repeats; the corrected analysis also shows monotone precision
>   gain with experience (88.3 to 95.1%) that the mixed-population version hid.
>
> The validation and test columns are unaffected. Cross-fitted figures are
> `fig_cf_ablation.png`, `fig_cf_difficult.png`, `fig_cf_selfimprove.png`,
> `fig_cf_gate.png`.

Branch `final-eval`. Configurations frozen in `reports/final_eval_plan.md`
(commit `b64ed9d`) **before any number here was computed**.

**Everything in this report is descriptive.** The project's confirmatory result
is Phase V's single test pass (`reports/phase5_memory.md`); that pass is spent.
Nothing here is tuned - every threshold, detector, policy and budget is taken as
already fixed in Phases III to V. Test columns are labelled descriptive and
support no hypothesis test, with the single exception of H4, whose verdict was
pre-registered on **out-of-fold** evidence with validation agreeing in sign.

M0 throughout is the Phase IIIb 5-fold ensemble.

The final agent: `topology_remove` as a fixed strategy, `p_failure >= 0.90`,
the Q gate, T = 10.

---

## 1. Section 21 - ablation

One component added at a time, every rung a separate configuration, all scored
through the same `CaseContext.score()`.

### Out-of-fold (n=738, 353 tumour-positive, 385 tumour-free)

| rung | Dice | tumour-free with FP | FP mL | FN mL | lesion sens | HD95 | imp/unch/deg % |
|---|---|---|---|---|---|---|---|
| Baseline (M0) | 0.6888 | 289/385 | 13.256 | 6.589 | 0.5454 | 111.29 | 0.0 / 100.0 / 0.0 |
| + uncertainty (0.90, frozen) | 0.6892 | 289/385 | 13.206 | 6.590 | 0.5452 | 109.14 | 0.8 / 99.2 / 0.0 |
| + uncertainty (matched rate) | 0.6919 | 268/385 | 11.977 | 6.782 | 0.5423 | 99.17 | 8.8 / 89.0 / 2.3 |
| + reasoning | 0.6919 | 249/385 | 12.009 | 6.589 | 0.5454 | 97.77 | 5.4 / 94.3 / 0.3 |
| + refinement (T=1, no Q gate) | 0.6918 | 251/385 | 11.984 | 6.589 | 0.5454 | 97.78 | 5.4 / 94.3 / 0.3 |
| + reflection (Q gate) | 0.6905 | 251/385 | 12.743 | 6.589 | 0.5454 | 99.18 | 4.0 / 95.8 / 0.3 |
| + iteration (T=10) | 0.6943 | 242/385 | 12.059 | 6.595 | 0.5452 | 94.58 | 6.5 / 93.2 / 0.3 |
| + learned strategy memory | 0.6913 | 246/385 | 12.974 | 6.589 | 0.5454 | 96.30 | 5.7 / 94.1 / 0.3 |
| **FINAL AGENT** | **0.6943** | **240/385** | 12.044 | 6.595 | 0.5452 | **94.58** | 6.5 / 93.2 / 0.3 |

### Validation (n=76, 38 tumour-positive, 38 tumour-free)

| rung | Dice | tumour-free with FP | FP mL | lesion sens | HD95 |
|---|---|---|---|---|---|
| Baseline | 0.7061 | 22/38 | 5.158 | 0.5878 | 79.95 |
| + uncertainty (0.90, frozen) | 0.7061 | 22/38 | 5.158 | 0.5878 | 79.95 |
| + uncertainty (matched rate) | 0.7061 | 21/38 | 5.148 | 0.5837 | 79.90 |
| + reasoning | 0.7068 | 20/38 | 5.067 | 0.5851 | 64.07 |
| + refinement | 0.7068 | 20/38 | 5.086 | 0.5851 | 64.08 |
| + reflection (Q gate) | 0.7065 | 20/38 | 5.096 | 0.5864 | 65.11 |
| + iteration (T=10) | 0.7065 | 20/38 | 5.096 | 0.5864 | 65.11 |
| + learned strategy memory | 0.7065 | 20/38 | 5.096 | 0.5864 | 65.11 |
| **FINAL AGENT** | **0.7065** | **20/38** | 5.086 | 0.5864 | **65.11** |

### Test (n=200, 110 tumour-positive, 90 tumour-free) - DESCRIPTIVE

| rung | Dice | tumour-free with FP | FP mL | lesion sens | HD95 |
|---|---|---|---|---|---|
| Baseline | 0.6638 | 54/90 | 9.084 | 0.5495 | 120.69 |
| + uncertainty (0.90, frozen) | 0.6638 | 54/90 | 9.084 | 0.5495 | 120.69 |
| + uncertainty (matched rate) | 0.6642 | 53/90 | 8.934 | 0.5462 | 119.53 |
| + reasoning | 0.6692 | 48/90 | 8.129 | 0.5484 | 93.30 |
| + refinement | 0.6690 | 48/90 | 8.222 | 0.5489 | 93.28 |
| + reflection (Q gate) | 0.6684 | 48/90 | 8.276 | 0.5489 | 98.37 |
| + iteration (T=10) | 0.6690 | 48/90 | 8.060 | 0.5484 | 98.54 |
| + learned strategy memory | 0.6696 | 48/90 | 8.820 | 0.5484 | 98.49 |
| **FINAL AGENT** | 0.6690 | 48/90 | **8.048** | 0.5484 | 98.55 |

Figure: `fig_final_ablation.png`.

### What the ladder says

**The frozen uncertainty rung is a near no-op, and that is the finding.** At the
pre-registered 0.90 gate the uncertainty-only detector fires on **34 of 6,462**
out-of-fold components; the reasoning detector fires on **663**. Uncertainty
alone is almost never 90% confident. The matched-rate row - the same detector at
threshold 0.7312, chosen so it acts on exactly 663 components too - is an
addition to the frozen plan and is labelled as such. It separates two different
claims: at a matched action count uncertainty does real work (289 -> 268
patients), so its limitation is **calibration**, not ranking. But reasoning
still clears **19 more tumour-free patients at the same action count**
(268 -> 249), and does it without the sensitivity cost the matched uncertainty
arm pays (0.5423 against 0.5454). That difference is what the reasoning features
buy.

**The Q gate is the only rung that moves Dice backwards.** 0.6918 -> 0.6905
out-of-fold, with FP volume rising 11.98 -> 12.74 mL. Phase IV falsified Q as a
judge of single actions at 37.1% precision against a 60% threshold; here the
same finding arrives by a different route, in a table that was built to be
neutral about it. The final agent keeps the gate because the work plan specifies
it, and because its cost is small and its presence makes the configuration
conservative - but the honest reading is that it is not earning its place.

**Iteration is the largest single step**, 0.6918 -> 0.6943 and 251 -> 242
patients. This matches Phase V, where raising T from 3 to 10 was also the
biggest lever.

**Learned strategy memory costs Dice**, 0.6943 -> 0.6913, and clears fewer
patients than the fixed strategy at the same budget. This is the third
independent confirmation of Phase V's H5 verdict, after the attribution
experiment and the self-improvement curve in section 4 below.

## 2. Section 22 - difficult cases, and H4

Baseline against the final agent. Tertile boundaries fixed **on out-of-fold**
(Q 0.130 / 0.680, `u_stab` 0.0382 / 0.0834) and applied unchanged elsewhere.

### Out-of-fold

| group | n | dDice |
|---|---|---|
| **Q low (hardest)** | 29 | **+0.0209 [+0.0003, +0.0512]** |
| Q mid | 158 | +0.0079 [+0.0025, +0.0175] |
| Q high (easiest) | 166 | +0.0006 [-0.0000, +0.0015] |
| **u_stab high (hardest)** | 48 | **+0.0162 [+0.0004, +0.0451]** |
| u_stab mid | 138 | +0.0070 [+0.0024, +0.0137] |
| u_stab low (easiest) | 164 | +0.0012 [+0.0004, +0.0023] |
| anatomically inconsistent | 30 | +0.0010 [+0.0003, +0.0019] |
| metabolically inconsistent | 19 | +0.0042 [+0.0002, +0.0109] |
| topologically inconsistent | 154 | +0.0008 [+0.0001, +0.0019] |

### Validation and test

| group | val | test (descriptive) |
|---|---|---|
| Q low (hardest) | +0.0057 [+0.0000, +0.0170] | +0.0345 [+0.0000, +0.1009] |
| Q high (easiest) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| u_stab high (hardest) | n=0, too few | +0.0099 [+0.0000, +0.0268] |
| u_stab low (easiest) | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0033, +0.0024] |

Figure: `fig_final_difficult.png`.

### Verdict: H4 is SUPPORTED

The pre-registered criterion was that the hardest tertile's dDice exceeds the
easiest tertile's **and** the hardest tertile's interval excludes zero on
out-of-fold, with validation agreeing in sign.

Both difficulty measures satisfy it. By Q, +0.0209 [+0.0003, +0.0512] against
+0.0006; by uncertainty, +0.0162 [+0.0004, +0.0451] against +0.0012. Validation
agrees in sign on the Q split (+0.0057 against +0.0000). The gradient is
monotone in both: hardest > mid > easiest.

**This reverses the verdict Phases IV and V reached.** Phase IV rejected H4 with
the wrong sign (-0.0544 on the hardest out-of-fold tertile) and Phase V with the
right sign but intervals including zero. The difference is the agent, not the
analysis: Phase IV's loop gated on Q, which sent it to act almost exclusively on
cases Q thought hopeless and do damage there. The final agent gates on
`p_failure` instead, and the cases where the detector is confident turn out to
be concentrated among the hard ones.

Two qualifications. The hardest-tertile intervals are wide and rest on 29 and 48
cases. And the three inconsistency groups show small but reliably positive
effects, not large ones - the agent helps a little on nearly all of them rather
than a lot on a few.

## 3. Section 23 - failure recovery, final agent

| split | improved | unchanged | degraded | mean dDice | mean dHD95 | iters/case | s/case |
|---|---|---|---|---|---|---|---|
| out-of-fold | 6.5% | 93.2% | **0.3%** | +0.00553 | -16.71 | 0.88 | 0.5 |
| validation | 2.6% | 94.7% | 2.6% | +0.00047 | -11.07 | 0.34 | 0.1 |
| test (descriptive) | 4.5% | 94.5% | **0.9%** | +0.00523 | -20.80 | 0.47 | 0.5 |

The asymmetry is the point: on out-of-fold the agent improves 6.5% of
tumour-positive cases and degrades 0.3%, a ratio of about twenty to one, and on
test 4.5% against 0.9%. It acts rarely - under one action per case - and when it
does, it is right far more often than not. HD95 falls by 11 to 21 mm on every
split, which is a larger relative movement than Dice and reflects that most of
what the agent removes is far from the tumour.

## 4. Section 24 - the self-improvement curve

Experience measured in out-of-fold patients, sampled so no patient is split
across sizes. Evaluated on the **gated** population - components with
`p_failure >= 0.90`, 677 components and 3,067 candidate actions - because that
is what the agent actually chooses among.

| experience | patients | actions seen | R(a) = E[dQ \| a] | mean d_qdice | precision |
|---|---|---|---|---|---|
| 10% | 58 | 5,982 | +0.03766 | +0.03771 | 96.7% |
| 25% | 144 | 10,911 | +0.03992 | +0.04156 | 93.6% |
| 50% | 289 | 18,355 | +0.03456 | +0.03788 | 96.5% |
| 100% (in-sample) | 578 | 34,999 | +0.03964 | +0.03956 | 92.2% |
| **fixed strategy (no experience)** | - | 0 | **+0.03946** | **+0.04287** | **98.0%** |

Figure: `fig_final_selfimprove.png`.

**There is no self-improvement.** The curve is flat within noise from 58 patients
to 578, and the **no-experience reference line sits at or above every point** -
highest precision (98.0%) and highest mean effect. Ten times more experience
buys nothing, because once the detector is 90% confident that a component is a
false positive, removing it is the right action 98% of the time and there is
almost nothing left for a selector to decide.

**The gating matters to this conclusion and is worth stating.** Computed over
all 34,999 actions rather than the gated 3,067, the fixed-strategy line falls to
-0.0178 mean d_qdice and 27.8% precision, and the learned selector appears to
beat it comfortably. That version would be wrong: it describes a selector
choosing among components the agent never reaches. Blanket removal is only
sensible once the detector is confident, which is exactly what the gate
guarantees. The first version of this table made that mistake.

This is the third independent line of evidence for the same conclusion, after
Phase V's attribution experiment and section 1's ablation.

## 5. Hypothesis summary

| | hypothesis | verdict | key number | evidence |
|---|---|---|---|---|
| **H1** | uncertainty correlates with segmentation error | **supported** | component-level AUROC 0.79-0.85; case mean uncertainty Spearman 0.74 against 1-Dice | `phase2_uncertainty.md` |
| **H2** | reasoning features add over uncertainty | **split by task** | removal: -0.0106, straddles zero, Part 1's positive finding **withdrawn** as a `z_rel` scan-edge confound. recovery: all four contrasts exclude zero (+0.0757 to +0.1492) | `phase3b_cv.md` |
| **H3** | refinement improves the initial prediction | **split by agent generation.** Not supported for the single-pass Phase IV agent; **supported, with a small effect**, for the iterative gated agent from Phase V onwards | Phase IV: pre-registered agent accepted 0 of 738 cases, ban-lifted +0.0005 [-0.0055, +0.0075] on test. Phase V (confirmatory): **+0.0067 [+0.0012, +0.0164]**, 0 degraded of 110. Phase VI (descriptive): +0.0149 [+0.0024, +0.0320] on nnU-Net f0 and +0.0098 [+0.0041, +0.0169] on Swin | `phase4_agent.md`, `phase5_memory.md`, `phase6_swin.md` |
| **H4** | gains are larger in difficult cases | **narrowly supported** | hardest Q tertile +0.0209 [**+0.0003**, +0.0512] against easiest +0.0006; by uncertainty +0.0162 [**+0.0004**, +0.0451]. Both lower bounds sit just above zero on 29 and 48 cases, so the direction is clear and the magnitude is not | this report, section 2 |
| **H5** | experience-based strategy memory improves selection | **not supported** | fixed strategy +0.0079 against learned +0.0057 out-of-fold at matched budget and gate; self-improvement curve flat | `phase5_memory.md` section 6a, this report section 4 |
| **H6** | the agent transfers to a backbone it was not built on | **supported** | on Swin test, dDice +0.0098 [+0.0041, +0.0169] and 78 -> 72 tumour-free patients with any FP, all four pre-registered criteria met; +0.0149 and 60 -> 50 on nnU-Net f0, so "helps on both, less on Swin" | `phase6_swin.md` |

**On H3's two generations.** The distinction is the loop, not the metric. The
Phase IV agent made a single gated pass and its accept/reject decision ran
through Q, which Phase IV then falsified at 37.1% precision; it accepted nothing
in the pre-registered configuration and nothing measurable in the ban-lifted one.
The Phase V agent onwards gates on the detector's own confidence, iterates to
T = 10, and produces a small but reliable gain. One caveat on the Phase VI
figures: +0.0149 is measured against the **single** nnU-Net fold-0 model, a
weaker baseline than the 5-fold ensemble M0 that Phases IV and V were measured
against, so it should not be read as a larger effect than Phase V's +0.0067 on
the same scale.

**On H4's margin.** Both lower bounds are within 0.0004 of zero. The gradient is
monotone across tertiles on both difficulty measures and validation agrees in
sign, which is why the verdict is support rather than a null - but a reader
should treat the size of the effect in hard cases as unestablished.

### The confirmatory result

Phase V's single test pass, which is the only confirmatory number in the
project:

| co-primary | M0 | Phase V agent | effect |
|---|---|---|---|
| Dice, 110 tumour-positive cases | 0.6638 | 0.6705 | +0.0067 [+0.0012, +0.0164] |
| tumour-free patients with any FP (n=90) | 54 | 48 | 6 cleared, 0 added, McNemar p = 0.031 |

Met both co-primaries under Holm with lesion sensitivity -0.0017 inside the
-0.01 margin, **with one documented deviation in selector choice**
(`phase5_memory.md` section 9). The configuration in that pass used the learned
selector; this report's section 1 shows the fixed strategy would have done
slightly better, which is the cost of that deviation.

### What the project establishes, in one paragraph

The loop also transfers: applied unchanged to a SwinUNETR backbone none of its
components were fitted on, it still meets both co-primaries (H6), though it helps
less there and the Q gate - the one component tuned to a particular model's
output shape - rejects 80% of proposals against 67% on nnU-Net. The Swin backbone
itself is 0.057 Dice worse than nnU-Net under matched conditions, and the agent
recovers about a sixth of that gap, so repair is not a substitute for a better
base model.

Uncertainty tracks error well enough to rank components (H1) but is too poorly
calibrated to gate on directly. Reasoning features do not beat uncertainty at
deciding what to remove, but do at deciding what to recover (H2). A
diagnose-and-repair loop over a frozen segmentation model produces a small,
reliable improvement - about +0.005 to +0.007 Dice and a sixth of tumour-free
patients cleared of false positives - but only once its accept/reject decision
is routed through the failure detector's own confidence rather than through a
learned quality critic (H3 failed with the critic, the final agent succeeds
without it). The gains concentrate in the hardest cases (H4). Learning which
strategy to apply adds nothing over always applying the same one (H5), because
the gate has already done the hard part of the decision.

## 6. Deviations from the frozen plan

1. **`use_checkpoint=False` in the Swin trainer** (2026-10-17). `build_swin_unetr`
   sets gradient checkpointing on; it was turned off for the nnU-Net trainer so
   the run fits the agreed wall-clock budget (25.2 h -> 20.8 h, peak memory
   11.6 -> 17.0 GB). Gradient checkpointing is exact recomputation, so gradients
   and the trained model are unchanged. Decided with the user before launch.
2. **The matched-rate uncertainty row** (2026-10-17) is an addition to section 21,
   not a replacement: the frozen rung 2 is reported as frozen. It was added
   because the frozen rung is a near no-op and would otherwise say nothing about
   ranking quality.
3. **Section 24 is computed on the gated population**, not on all 34,999 measured
   actions. The ungated version describes a selector choosing among components
   the agent never reaches, and reverses the conclusion. Stated in section 4.
