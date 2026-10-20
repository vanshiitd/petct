# Results for the thesis - every number and figure in one place

Single reference for writing up. Every number below is copied from the JSON the
analysis scripts wrote, not retyped from a report. Intervals are 95% paired
bootstrap over patients, 2000 resamples, unless the line says McNemar.

**Read this first.** The project has exactly **one confirmatory result**: Phase
V's single test pass (table 2). Everything else is descriptive or pre-registered
on out-of-fold data. Where a table is descriptive it says so, and no p-value in
a descriptive table should be reported as a hypothesis test.

Repository: https://github.com/vanshiitd/petct, branch `main` (PRs #1-#5 merged).

**This directory (`docs/results/`) holds everything the write-up needs**: this
document, every phase report, every figure, and copies of the three
pre-registrations. All filenames below are relative to this directory, so a
link to `phase4_agent.md` or `fig_final_ablation.png` resolves here.

Two things are deliberately **not** committed, since the repository carries no
data, predictions, checkpoints or logs:

- **The analysis JSON** each table was built from (`final_eval_tables.json` and
  the rest). Every table below names the script that regenerates it, and the
  numbers here were copied from those files rather than retyped.
- **The maps, candidate CSVs and per-case outputs** the scripts read. Those
  live outside the repository under `D:\data`.

The pre-registration copies here are for convenience. The authoritative ones
are `reports/phase4_analysis_plan.md`, `reports/phase5_analysis_plan.md` and
`reports/final_eval_plan.md` at the repository root, whose commits are what
establish that each was written before the numbers it governs.

---

## 0. Definitions used throughout

| term | definition |
|---|---|
| M0 | the Phase IIIb 5-fold ensemble. Test Dice 0.6638 |
| Dice | mean over **tumour-positive** cases only |
| FP / FN volume | mean over **all** cases, mL |
| tumour-free with any FP | count of tumour-free patients having at least one false-positive component |
| lesion sensitivity | GT lesions touched by any predicted component, pooled |
| HD95 | tumour-positive cases only, mm |
| improved / degraded | per case, `dDice > +0.01` / `< -0.01` |
| final agent | fixed `topology_remove`, `p_failure >= 0.90`, Q gate, T = 10 |

Splits: out-of-fold 738 scans (353 tumour-positive), validation 76 (38),
test 200 patients (110 tumour-positive, 90 tumour-free).

---

## Table 1 - Hypotheses and verdicts

| | hypothesis | verdict | key number |
|---|---|---|---|
| H1 | uncertainty correlates with segmentation error | **supported** | component-level AUROC 0.79-0.85; case mean uncertainty Spearman 0.74 vs 1-Dice |
| H2 | reasoning features add over uncertainty | **split by task** | removal -0.0106, straddles zero (Part 1 finding **withdrawn**, `z_rel` confound); recovery all four contrasts exclude zero, +0.0757 to +0.1492 |
| H3 | refinement improves the initial prediction | **split by agent generation** | single-pass Phase IV agent: **not supported**, +0.0005 [-0.0055, +0.0075]. Iterative gated agent, Phase V on: **supported, small effect**, +0.0067 [+0.0012, +0.0164] |
| H4 | gains are larger in difficult cases | **narrowly supported** | hardest Q tertile +0.0209 [**+0.0003**, +0.0512] vs easiest +0.0006; by uncertainty +0.0162 [**+0.0004**, +0.0451] |
| H5 | experience-based strategy memory improves selection | **not supported** | fixed strategy +0.0079 vs learned +0.0057 out-of-fold at matched budget and gate; self-improvement curve flat |
| H6 | the agent transfers to a backbone it was not built on | **supported** | Swin test +0.0098 [+0.0041, +0.0169], 78 -> 72 patients cleared; all four pre-registered criteria met |

Sources: `phase2_uncertainty.md`, `phase3b_cv.md`, `phase4_agent.md`,
`phase5_memory.md`, `final_eval.md`, `phase6_swin.md`.

**H3's two generations.** Phase IV made a single gated pass whose accept/reject
decision ran through Q, which the same phase falsified at 37.1% precision
against a 60% threshold; its pre-registered configuration accepted **0 actions
in 738 cases**. From Phase V the loop gates on the detector's own confidence and
iterates to T = 10, and the effect becomes real but small.

**H4's margin.** Both lower bounds are within 0.0004 of zero, on 29 and 48
cases. The gradient is monotone across tertiles on both difficulty measures and
validation agrees in sign, so the direction is established and the magnitude is
not.

---

## Table 2 - THE CONFIRMATORY RESULT (Phase V, single test pass)

Pre-registered in `phase5_analysis_plan.md` (`8dc2fea`), test read once. Configuration: learned action-value selector, `p_failure >= 0.90`, T = 10,
no `roi_repredict`, no Q gate.

| co-primary | M0 | Phase V agent | effect |
|---|---|---|---|
| Dice, 110 tumour-positive | 0.6638 | 0.6705 | **+0.0067 [+0.0012, +0.0164]**, p < 0.001 |
| tumour-free with any FP, n=90 | 54 | **48** | 6 cleared, 0 added, exact McNemar **p = 0.031** |

Safety: lesion sensitivity 0.5495 -> 0.5478, **-0.0017**, inside the -0.01
margin. Both co-primaries superior under Holm at family-wise alpha 0.05. Outcome
split: improved 9, unchanged 101, **degraded 0**.

**Caveats to state in the text.** The effect is small - about a third of what
ensembling five models bought in Phase IIIb (+0.0213). Co-primary 2 rests on 6
discordant pairs, so p = 0.031 is the smallest value six pairs can produce and
it clears its Holm threshold only because Dice took the stricter slot; the
verdict rests on Dice. And the configuration carried one **documented deviation
in selector choice** (`phase5_memory.md` section 9): the learned selector was
used although the pre-registered precision check rejected it, a decision made on
out-of-fold and validation evidence 17 hours before test was read. Table 5 shows
the fixed strategy would have done slightly better.

Source: `phase5_memory.md` section 7; regenerate with
`scripts/phase5_eval.py --agent p5_test_final --ref p5_test_final`.

---

## Table 3 - Section 21 ablation (descriptive)

One component added at a time from M0, every rung a separate configuration, all
scored through one metric implementation.

### Out-of-fold (n=738)

| rung | Dice | tumour-free with FP | FP mL | FN mL | lesion sens | HD95 | imp/unch/deg % |
|---|---|---|---|---|---|---|---|
| Baseline (M0) | 0.6888 | 289/385 | 13.256 | 6.589 | 0.5454 | 111.29 | 0.0 / 100.0 / 0.0 |
| + uncertainty (0.90, frozen) | 0.6892 | 289/385 | 13.206 | 6.590 | 0.5452 | 109.14 | 0.8 / 99.2 / 0.0 |
| + uncertainty (matched rate) | 0.6919 | 268/385 | 11.977 | 6.782 | 0.5423 | 99.17 | 8.8 / 89.0 / 2.3 |
| + reasoning | 0.6919 | 249/385 | 12.009 | 6.589 | 0.5454 | 97.77 | 5.4 / 94.3 / 0.3 |
| + refinement (T=1, no Q gate) | 0.6918 | 251/385 | 11.984 | 6.589 | 0.5454 | 97.78 | 5.4 / 94.3 / 0.3 |
| + reflection (Q gate) | **0.6905** | 251/385 | 12.743 | 6.589 | 0.5454 | 99.18 | 4.0 / 95.8 / 0.3 |
| + iteration (T=10) | 0.6943 | 242/385 | 12.059 | 6.595 | 0.5452 | 94.58 | 6.5 / 93.2 / 0.3 |
| + learned strategy memory | 0.6913 | 246/385 | 12.974 | 6.589 | 0.5454 | 96.30 | 5.7 / 94.1 / 0.3 |
| **FINAL AGENT** | **0.6943** | **240/385** | 12.044 | 6.595 | 0.5452 | **94.58** | 6.5 / 93.2 / 0.3 |

### Test (n=200, descriptive)

| rung | Dice | tumour-free with FP | FP mL | lesion sens | HD95 |
|---|---|---|---|---|---|
| Baseline (M0) | 0.6638 | 54/90 | 9.084 | 0.5495 | 120.69 |
| + uncertainty (0.90, frozen) | 0.6638 | 54/90 | 9.084 | 0.5495 | 120.69 |
| + uncertainty (matched rate) | 0.6642 | 53/90 | 8.934 | 0.5462 | 119.53 |
| + reasoning | 0.6692 | 48/90 | 8.129 | 0.5484 | 93.30 |
| + refinement | 0.6690 | 48/90 | 8.222 | 0.5489 | 93.28 |
| + reflection (Q gate) | 0.6684 | 48/90 | 8.276 | 0.5489 | 98.37 |
| + iteration (T=10) | 0.6690 | 48/90 | 8.060 | 0.5484 | 98.54 |
| + learned strategy memory | 0.6696 | 48/90 | 8.820 | 0.5484 | 98.49 |
| **FINAL AGENT** | 0.6690 | 48/90 | **8.048** | 0.5484 | 98.55 |

Validation table in `final_eval.md` section 1.

**Three points for the text.** The frozen uncertainty rung is a near no-op
because the uncertainty-only detector reaches 0.90 on **34 of 6,462**
out-of-fold components against the reasoning detector's **663** - uncertainty
alone is almost never 90% confident. The matched-rate row (threshold 0.7312,
chosen so it acts on exactly 663 too) is an **addition to the frozen plan**, not
a replacement, and shows the limitation is calibration rather than ranking;
reasoning still clears 19 more patients at the same action count and without the
sensitivity cost. **The Q gate is the only rung that moves Dice backwards.**

Figure: `fig_final_ablation.png`. Full tables in `final_eval.md` section 1;
regenerate with `scripts/final_eval_tables.py`.

---

## Table 4 - Section 22 difficult cases, and H4 (out-of-fold)

Tertile boundaries fixed on out-of-fold (Q 0.130 / 0.680, `u_stab` 0.0382 /
0.0834) and applied unchanged elsewhere. Baseline vs the final agent.

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

Test, descriptive: Q low +0.0345 [+0.0000, +0.1009], Q high +0.0000.

**This reverses Phases IV and V**, and the difference is the agent rather than
the analysis: Phase IV gated on Q, which sent it to act almost only on cases Q
thought hopeless and do damage there (-0.0544 on the hardest tertile). The final
agent gates on `p_failure`, and confident cases concentrate among the hard ones.

Figure: `fig_final_difficult.png`.

---

## Table 5 - H5, attribution at matched budget and gate (out-of-fold)

All arms at T = 10, `p_failure >= 0.90`, varying only how the strategy is chosen.

| arm | dDice | tumour-free with FP | dFP mL |
|---|---|---|---|
| **(a) fixed `topology_remove`, no selection** | **+0.0079 [+0.0036, +0.0131]** | 289 -> **229** | **-2.341** |
| **(b) rule-based policy** | **+0.0079 [+0.0036, +0.0130]** | 289 -> **231** | **-2.308** |
| (c) learned selector | +0.0057 [+0.0025, +0.0102] | 289 -> 235 | -0.607 |

Validation: all three identical, +0.0008 [-0.0010, +0.0030], 22 -> 20.

**(a) and (b) are indistinguishable and both beat (c).** The learned selector
picks `topology_remove` for 495 of 499 actions, so it approximates a fixed
strategy and pays an approximation error for it. No test number was computed for
(a) or (b): test had already been used once.

### Self-improvement curve (section 24), gated population

677 components, 3,067 candidate actions at `p_failure >= 0.90`.

| experience | patients | R(a) = E[dQ \| a] | mean d_qdice | precision |
|---|---|---|---|---|
| 10% | 58 | +0.03766 | +0.03771 | 96.7% |
| 25% | 144 | +0.03992 | +0.04156 | 93.6% |
| 50% | 289 | +0.03456 | +0.03788 | 96.5% |
| 100% (in-sample) | 578 | +0.03964 | +0.03956 | 92.2% |
| **fixed strategy (no experience)** | - | **+0.03946** | **+0.04287** | **98.0%** |

Flat within noise, with the no-experience line at or above every point.

**Methodological note for the text.** Computed over all 34,999 measured actions
instead of the gated 3,067, the fixed line falls to -0.0178 and 27.8% and the
learned selector appears to win. That version is wrong: it describes a selector
choosing among components the agent never reaches.

Figure: `fig_final_selfimprove.png`. Attribution in `phase5_memory.md`
section 6a; regenerate with `scripts/final_selfimprove.py`.

---

## Table 6 - Q is not a usable judge of single actions (Phase IV)

| strategy | n | dQ>0 | honest | gamed | precision |
|---|---|---|---|---|---|
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

**37.1% against a pre-registered 60% threshold; no strategy clears it.**
Spearman(dQ, dDice) = 0.0745. The median action moves Dice by 0.00021, two to
three orders of magnitude below Q's own fit error (R-squared 0.6305). Q ranks
whole masks well (within-case Spearman 0.6966) - ranking a mask and adjudicating
one edit to it are different jobs.

The compact illustration, two out-of-fold cases with near-identical features and
opposite outcomes:

```
PETCT_0e2034240b       p_fail=0.579  removes 54 mL   Dice 0.5329 -> 0.0000   dQ=-0.2859
PETCT_fde79b6aa9_...   p_fail=0.967  removes 188 mL of pure FP, nothing true lost  dQ=-0.2029
```

Q scored the single most valuable action in the run **more negatively** than the
second-worst. The signal that separates them is the detector's own confidence.

Figure: `fig_pfail_guard.png`. Source: `phase4_agent.md` sections 4 and 10.

---

## Table 7 - Phase VI, backbone comparison (Experiment 1)

SwinUNETR-small, MAE pre-trained, against the **single** nnU-Net fold-0 model.
Matched dataset, plans, split, epochs, preprocessing and 8-flip fusion. 167 of
167 tensors transferred, 0 missing.

| metric | split | nnU-Net f0 | Swin | delta |
|---|---|---|---|---|
| Dice | val | 0.6959 | 0.6594 | **-0.0365 [-0.0611, -0.0140]** |
| Dice | test | 0.6425 | 0.5860 | **-0.0565 [-0.0866, -0.0280]** |
| FP volume mL | test | 15.934 | 23.790 | **+7.856 [+1.139, +15.361]** |
| FN volume mL | test | 5.372 | 5.391 | +0.018 [-1.097, +1.098] |
| HD95 | test | 128.27 | 188.22 | **+59.95 [+29.28, +94.95]** |
| lesion sensitivity | test | 0.5517 | 0.5178 | -0.0339 |
| tumour-free with any FP | test | 60/90 | **78/90** | +18 |

**The convolutional backbone wins at this data scale**, and the failure mode is
specific: FN volume is unchanged, so Swin is not missing more tumour - it is
drawing more things that are not tumour.

Full detail in `phase6_swin.md` section 2; regenerate with
`scripts/phase6_compare.py`.

---

## Table 8 - Phase VI, H6 transfer (pre-registered)

Final agent applied **unchanged** - Q model, detector, thresholds, budget, all
fitted on nnU-Net out-of-fold data.

| criterion (frozen in `final_eval_plan.md` section 6) | required | Swin test | met |
|---|---|---|---|
| dDice non-inferior | lower bound > -0.01 | **+0.0098 [+0.0041, +0.0169]** | yes |
| tumour-free FP count not increased | <= 78 | **72** (6 cleared, 0 added) | yes |
| lesion sensitivity non-inferior | >= -0.01 | **-0.0017** | yes |
| at least one co-primary improves | >= 1 | both | yes |

| | on Swin (test) | on nnU-Net f0 (test) |
|---|---|---|
| Dice | 0.5860 -> 0.5958, +0.0098 [+0.0041, +0.0169] | 0.6425 -> 0.6574, +0.0149 [+0.0024, +0.0320] |
| tumour-free with any FP | 78 -> 72, p = 0.031 | 60 -> 50, p = 0.0020 |
| FP volume mL | -3.193 [-5.945, -1.252] | -1.713 [-3.390, -0.520] |
| HD95 | -4.94 [-13.83, +0.58] | -15.92 [-33.02, -3.93] |
| lesion sensitivity | -0.0017 | -0.0028 |
| improved / unchanged / degraded | 12 / 97 / **1** | 9 / 100 / **1** |

**"Helps on both, less on Swin."** Note the nnU-Net f0 column is against the
**single** fold-0 model, a weaker baseline than M0, so +0.0149 is not a larger
effect than table 2's +0.0067 on the same scale.

### The gate rejection rate - the most informative number in Phase VI

| | attempts | accepted | rejected by the Q gate |
|---|---|---|---|
| Swin, test | 460 | 90 | **80%** |
| nnU-Net f0, test | 177 | 58 | 67% |
| Swin, val | 157 | 23 | **85%** |
| nnU-Net f0, val | 52 | 13 | 75% |

The detector proposes far more on Swin (460 vs 177) because there are more false
positives to find, and the Q model - fitted on nnU-Net masks - is systematically
less willing to approve edits to Swin masks. Q is the only component in the loop
fitted to the shape of a particular model's output rather than to anatomy.

**The agent recovers about 17% of the backbone gap**: Swin with the agent 0.5958,
nnU-Net without it 0.6425.

Figure: `fig_phase6_transfer.png`. Full detail in `phase6_swin.md`
section 3.

---

## Table 9 - Failure diagnosis and oracle headroom (Phase IV, out-of-fold)

What the base model gets wrong, and whether it can be spotted without ground
truth. 353 tumour-positive cases, mean Dice 0.6888.

| failure class | items / cases | oracle dDice | detectability (CV / val AUROC) |
|---|---|---|---|
| missed | 3347 / 263 | **+0.1259** | **0.6470 / 0.7305** |
| under_seg | 431 / 122 | +0.0950 | 0.8009 / 0.7731 |
| boundary | 573 / 205 | +0.0945 | 0.8245 / 0.7870 |
| over_seg | 464 / 90 | +0.0701 | 0.8051 / 0.8492 |
| spurious_other | 2490 / 284 | +0.0538 | 0.8432 / 0.8805 |
| spurious_liver_spleen | 54 / 20 | +0.0146 | 0.9646 / 0.9968 |
| spurious_physio | 37 / 13 | +0.0021 | 0.9668 / 0.9992 |
| fragmentation | 253 / 113 | topology only | 0.7076 / 0.6925 |
| merging | 152 / 87 | topology only | 0.8914 / 0.8837 |

**The central tension of the project in one table: the detectable classes are
worth almost nothing, and the valuable class is the least detectable.**

Source: `phase4_agent.md` sections 1 and 2.

---

## Figures

| file | shows | table |
|---|---|---|
| `fig_final_ablation.png` | section 21 ablation, Dice and FP burden, three splits | 3 |
| `fig_final_difficult.png` | dDice by difficulty group with intervals (H4) | 4 |
| `fig_final_selfimprove.png` | self-improvement curve against the no-experience line | 5 |
| `fig_pfail_guard.png` | dDice and lesions lost by `p_failure` bin | 6 |
| `fig_phase6_transfer.png` | backbone comparison, agent effect, FP clearance | 7, 8 |
| `fig_strategy_heatmap_s6.png` | mean dDice of every strategy on every failure class | 9 |
| `fig_ablation_ladder.png` | Phase V iteration ladder, guarded vs unguarded | - |

All in this directory.

---

## Pre-registrations, in the repository

| file | commit | fixes |
|---|---|---|
| `phase4_analysis_plan.md` (authoritative: `reports/`) | `cfb6756` | Phase IV endpoints, safety margin, gaming check, falsification |
| `phase5_analysis_plan.md` (authoritative: `reports/`) | `8dc2fea` | co-primary endpoints, Holm, McNemar, H5 |
| `final_eval_plan.md` (authoritative: `reports/`) | `b64ed9d` | sections 21-24 configurations, H6 criteria |

Each was committed before any number it governs was computed.

---

## Deviations and corrections to disclose

1. **Phase V selector choice.** The learned selector was carried to test although
   the pre-registered precision check rejected it, decided on out-of-fold and
   validation evidence about 17 hours before test was read. Table 5 shows the
   fixed strategy would have done slightly better. `phase5_memory.md` section 9.
2. **Phase V's criterion was misspecified.** It ranked selectors by how often
   they help rather than how much; the post-hoc sign classifier wins precision
   (59.4% vs 47.4%) while picking actions worth an eighth as much.
3. **H2's removal finding was withdrawn** between Phase III Part 1 and Part 2:
   `z_rel` was a scan-edge indicator, not an anatomical prior, and falls below
   chance (test AUROC 0.4665) once its extreme deciles are trimmed.
4. **Two leaks found and fixed in Phase IIIb**: cross-validation grouped by scan
   rather than patient (108 of 738 scans shared a patient across folds), and GBT
   hyperparameters chosen while fitting on validation.
5. **`use_checkpoint=False`** in the Swin trainer, to fit the wall-clock budget.
   Exact recomputation, so the trained model is unchanged.
6. **The matched-rate uncertainty row** in table 3 is an addition to the frozen
   plan, not a replacement for the frozen rung.
7. **Section 24 is computed on the gated population.** The ungated version
   reverses the conclusion and is wrong; both are reported.
8. **`roi_repredict` is falsified** at both T = 3 and T = 10, and is not in the
   final agent.

---

## One-paragraph summary

Uncertainty tracks segmentation error well enough to rank components (H1) but is
too poorly calibrated to gate on directly - it reaches 90% confidence on 0.5% of
components. Reasoning features do not beat uncertainty at deciding what to
remove, but do at deciding what to recover (H2). A diagnose-and-repair loop over
a frozen segmentation model produces no measurable gain when its accept/reject
decision runs through a learned quality critic (H3, Phase IV), and a small,
reliable gain once it runs through the failure detector's own confidence and is
allowed to iterate - **+0.0067 Dice [+0.0012, +0.0164] and 6 of 90 tumour-free
patients cleared of false positives, with no case degraded** (H3, Phase V). The
gains concentrate in the hardest cases, narrowly (H4). Learning which strategy
to apply adds nothing over always applying the same one (H5), because the
confidence gate has already made the hard part of the decision. The loop
transfers unchanged to a transformer backbone it was never fitted on (H6),
though it helps less there, and that backbone is itself 0.057 Dice worse than
nnU-Net - repair does not substitute for a better base model.
