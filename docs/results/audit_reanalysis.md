# External-examiner audit: re-analysis

Branch `audit-reanalysis`. Every number here is new; no existing report, figure,
JSON or prediction was modified. Nothing was tuned or chosen on test: thresholds
were fixed out-of-fold, checked on validation, and test numbers are either taken
from existing runs or computed once and labelled **descriptive**.

Each section names the script and the JSON behind its numbers.

---

## Summary: what changes

| finding | status |
|---|---|
| **The "98%" gated precision is in-sample.** Cross-fitted it is **85.3%**, and lesions lost at the 0.90 gate rise from 3 to 41 | **contradicts `final_eval.md` §4** |
| **The +reasoning ablation rung loses its Dice gain.** Published +0.0031, cross-fitted **-0.0012 [-0.0064, +0.0025]** | **contradicts `final_eval.md` §1** |
| **H4 does not survive multiplicity correction.** Both hardest-group Bonferroni lower bounds fall to **+0.0000** | **weakens `final_eval.md` §2** |
| **"A fixed strategy matches or beats the learned selector" is false cross-fitted.** Fixed +0.0033 (straddles zero), learned +0.0052, rule-based +0.0065 | **contradicts `phase5_memory.md` §6a** |
| **The post-fix uptake medians do not reproduce.** Report says 6.14x / 1.46x; recomputation gives **3.81 / 0.89** | **contradicts `nnunet_505_corrected_labels.md` l.49-50** |
| **The Phase IV guarded TEST run predates the Phase V pre-registration by ~22 h**, so the plan's "pre-registered secondary comparison" was written with those test numbers already in existence | **weakens `phase5_analysis_plan.md` §4** |
| The final agent itself survives cross-fitting: 0.6943 -> **0.6938**, 240 -> **239** patients cleared | holds |
| H5's verdict (learned does not beat the hand-built policy) | holds |
| Phase V's confirmatory test result | unchanged, and not re-run |
| U_stab beats U_ent at component level | holds, now with intervals |
| The simple volume baseline clears **zero** tumour-free patients at any threshold | holds, strongly favours the agent |

**No verdict flips outright.** H4 weakens from "narrowly supported" to "not
supported under correction"; H5's conclusion stands but the reasoning the thesis
gives for it does not.

---

## R1. Cross-fitted detectors

`scripts/audit_crossfit.py` -> `audit_crossfit.npz`, `audit_crossfit.json`.

`phase4_diag_model.py` l.91-94 keeps only `models[cname] = full`, fitted on all
738 out-of-fold scans, and `final_eval_detectors.py` l.63-65 does the same. The
agents then apply those models to the same scans. Q is the exception: it carries
per-fold models and the agents pick the fold that did not see the patient.

Refitted against the Phase IIIb patient-grouped folds 3-7, which partition the
738 scans exactly, hyperparameters unchanged:

| head | prevalence | in-sample AUROC | cross-fitted AUROC |
|---|---|---|---|
| spurious_physio | 0.0057 | **1.000** | 0.9994 |
| spurious_liver_spleen | 0.0084 | **1.000** | 0.9631 |
| spurious_other | 0.3853 | 0.9278 | 0.8179 |
| merging | 0.0235 | 0.9980 | 0.8832 |
| under_seg | 0.0806 | 0.9504 | 0.7930 |
| over_seg | 0.0340 | 0.9910 | 0.7218 |
| boundary | 0.1145 | 0.9396 | 0.8182 |
| fragmentation | 0.1082 | 0.9182 | 0.7155 |
| **recovery (missed)** | - | **0.9936** | **0.6082** |
| uncertainty-only | - | 0.8380 | 0.7533 |

**The AUROCs printed in `phase4_agent.md` §2 were already cross-validated and
survive** - 0.9668, 0.9646, 0.8432, 0.8914, 0.8009, 0.8051, 0.8245, 0.7076
against the cross-fitted column above, same range. What was in-sample is the
`p_failure` the agent gated on, and the recovery head is the worst case: 0.9936
in-sample against 0.6082 honest.

### R1(a) The gate, in-sample against cross-fitted

`scripts/audit_gate_table.py` -> `audit_gate_table.json`. Recomputed from the
measured strategy table, which for a fixed strategy is exactly equivalent to
re-running the agent.

| bin | n (IS) | mean dDice (IS) | lost (IS) | n (CF) | mean dDice (CF) | lost (CF) |
|---|---|---|---|---|---|---|
| < 0.60 | 4362 | -0.04149 | 3000 | 4344 | -0.04373 | 2981 |
| 0.60-0.80 | 1080 | -0.03135 | 484 | 1146 | -0.02486 | 421 |
| 0.80-0.90 | 357 | -0.01651 | 43 | 342 | -0.00572 | 87 |
| **0.90-0.95** | 120 | **+0.01038** | **3** | 118 | **-0.00339** | **26** |
| >= 0.95 | 543 | +0.00672 | 0 | 512 | +0.00378 | 15 |

| threshold | kept (IS) | mean dDice (IS) | lost (IS) | precision (IS) | kept (CF) | mean dDice (CF) | lost (CF) | precision (CF) |
|---|---|---|---|---|---|---|---|---|
| >= 0.60 | 2100 | -0.01885 | 530 | 69.6% | 2118 | -0.01534 | 549 | 53.1% |
| >= 0.80 | 1020 | -0.00073 | 46 | 90.7% | 972 | -0.00078 | 128 | 72.4% |
| **>= 0.90** | 663 | +0.00735 | **3** | **98.0%** | 630 | +0.00236 | **41** | **85.3%** |
| >= 0.95 | 543 | +0.00672 | 0 | 99.7% | 512 | +0.00378 | 15 | 93.0% |

The in-sample 98.0% reproduces `final_eval.md` §4 exactly, which confirms the
comparison is like-for-like. **Cross-fitted it is 85.3%, and the 0.90-0.95 band
flips from +0.01038 to -0.00339.**

**Is 0.90 still a reasonable gate?** As a *per-action* rule, no - cross-fitted,
the 0.90-0.95 band is harmful and the honest per-action threshold is >= 0.95.
But **at case level inside the loop, 0.90 remains the better setting**, because
the Q gate filters the marginal actions that the per-action table counts:
the final agent at 0.90 reaches +0.0050 and 239 patients cleared, against
+0.0037 and 248 at 0.95 (section R1(d) table). Both facts belong in the thesis;
quoting only the first would be as misleading as quoting only the second.

Gate counts: the published "34 vs 663" becomes **40 vs 630** cross-fitted. The
matched-rate uncertainty threshold moves 0.7312 -> **0.7420**.

### R1(b) Section 21 ablation, cross-fitted

`scripts/phase5_eval.py` -> `audit_eval_ablation.json`. Runs `au_oof_*`.

| rung | published Dice | cross-fitted Dice | dDice (CF) | negFP published | negFP CF |
|---|---|---|---|---|---|
| Baseline (M0) | 0.6888 | 0.6888 | - | 289/385 | 289/385 |
| + uncertainty (0.90) | 0.6892 | 0.6890 | +0.0002 [+0.0000, +0.0004] | 289 | 289 |
| + uncertainty (matched) | 0.6919 | 0.6874 | **-0.0014 [-0.0050, +0.0018]** | 268 | 274 |
| **+ reasoning** | **0.6919** | **0.6876** | **-0.0012 [-0.0064, +0.0025]** | 249 | 249 |
| + refinement | 0.6918 | 0.6908 | +0.0020 [+0.0012, +0.0029] | 251 | 250 |
| + reflection (Q gate) | 0.6905 | 0.6903 | +0.0015 [+0.0007, +0.0024] | 251 | 250 |
| + iteration (T=10) | 0.6943 | 0.6938 | +0.0050 [+0.0015, +0.0097] | 242 | 240 |
| + learned memory | 0.6913 | 0.6908 | +0.0020 [+0.0010, +0.0033] | 246 | 245 |
| **FINAL AGENT** | **0.6943** | **0.6938** | **+0.0050 [+0.0015, +0.0097]** | 240 | **239** |
| gate 0.95 (new) | - | 0.6925 | +0.0037 [+0.0011, +0.0078] | - | 248 |

**The single-pass rungs lose their Dice gain; the iterative ones do not.** The
final agent is essentially unchanged. The published claim that reasoning and
matched-rate uncertainty tie at 0.6919 becomes -0.0012 against -0.0014, both
straddling zero.

What survives of `final_eval.md` §1: reasoning still clears far more tumour-free
patients than matched uncertainty (**40 against 15** cleared, 249 against 274
remaining) and still at lower sensitivity cost (-0.0013 against -0.0053). The
false-positive argument for reasoning holds; the Dice argument does not.

One in-sample artefact disappears: the Q gate was "the only rung that moves Dice
backwards" (0.6918 -> 0.6905). Cross-fitted it is 0.6908 -> 0.6903, which is
within noise.

### R1(c) H4, with Bonferroni

`scripts/audit_h4_attr.py` -> `audit_h4_attr.json`. Boundaries kept at Q 0.130 /
0.680 and u_stab 0.0382 / 0.0834.

| group | n | dDice, 95% | Bonferroni 97.5% |
|---|---|---|---|
| **Q low (hardest)** | 29 | +0.0209 [+0.0003, +0.0512] | **[+0.0000, +0.0595]** |
| Q mid | 159 | +0.0071 [+0.0020, +0.0163] | |
| Q high (easiest) | 165 | +0.0002 [-0.0004, +0.0011] | |
| u_stab low (easiest) | 164 | +0.0007 [+0.0003, +0.0014] | |
| u_stab mid | 138 | +0.0064 [+0.0018, +0.0133] | |
| **u_stab high (hardest)** | 48 | +0.0162 [+0.0003, +0.0451] | **[+0.0000, +0.0557]** |
| anatomically inconsistent | 30 | +0.0005 [+0.0000, +0.0013] | |
| metabolically inconsistent | 19 | +0.0011 [-0.0000, +0.0025] | |
| topologically inconsistent | 154 | +0.0008 [+0.0001, +0.0018] | |

Validation: Q low (hardest) n=6, +0.0057 [+0.0000, +0.0170]; Q high n=14,
+0.0000. **u_stab high (hardest) has n=0 on validation**, so that arm cannot be
checked there at all.

**H4 survives cross-fitting at 95% but not multiplicity correction.** Both
hardest-group Bonferroni lower bounds are **+0.0000**. The monotone gradient
across tertiles is real and reproduces cross-fitted; the claim that the
hardest-group effect is distinguishable from zero does not survive correcting
for having tested two difficulty measures.

The three inconsistency groups are all positive but tiny (+0.0005 to +0.0011),
and the metabolic group straddles zero.

**Recommended wording:** H4 is directionally supported - gains are concentrated
in hard cases, monotonically, on two independent difficulty measures - but the
magnitude in the hardest group is not established at corrected significance.

### R1(d) Attribution, and R1(f)

All arms at T=10, `p_failure` >= 0.90, cross-fitted.

| configuration | dDice | 95% CI | negFP | dSens | acts/case | imp/unch/deg % |
|---|---|---|---|---|---|---|
| **(a) fixed** | **+0.0033** | **[-0.0035, +0.0099]** | 230/385 | -0.0034 | 0.82 | 7.4 / 91.2 / 1.4 |
| **(b) rule-based** | **+0.0065** | **[+0.0025, +0.0118]** | 231/385 | -0.0034 | 0.80 | 7.4 / 91.8 / 0.8 |
| **(c) learned** | **+0.0052** | **[+0.0022, +0.0099]** | 237/385 | -0.0035 | 0.77 | 7.1 / 92.4 / 0.6 |
| PV rung 2 (fixed, T=3) | -0.0002 | [-0.0056, +0.0038] | 237/385 | -0.0025 | 0.69 | 6.5 / 92.1 / 1.4 |
| PV rung 3 (learned, T=3) | +0.0028 | [+0.0016, +0.0043] | 242/385 | -0.0025 | 0.65 | 6.2 / 93.2 / 0.6 |
| PV rung 5-noroi (T=10) | +0.0052 | [+0.0022, +0.0099] | 237/385 | -0.0035 | 0.77 | 7.1 / 92.4 / 0.6 |
| R2(3) corrected variant | +0.0049 | [+0.0014, +0.0094] | 240/385 | -0.0006 | 0.26 | 4.8 / 94.6 / 0.6 |
| FINAL AGENT | +0.0050 | [+0.0015, +0.0097] | 239/385 | -0.0006 | 0.27 | 5.4 / 94.1 / 0.6 |
| gate 0.95 | +0.0037 | [+0.0011, +0.0078] | 248/385 | -0.0001 | 0.22 | 4.2 / 95.5 / 0.3 |

**This reverses the ordering the thesis reports.** In-sample, fixed (+0.0079)
tied rule-based (+0.0079) and both beat learned (+0.0057). Cross-fitted:
**rule-based (+0.0065) > learned (+0.0052) > fixed (+0.0033, straddling zero)**.

So:

- **H5 itself still fails.** The learned selector does not beat the hand-built
  policy (+0.0052 against +0.0065), which is what H5 claims.
- **But the argument the thesis uses for it is wrong.** `phase5_memory.md` §6a
  concludes "a fixed strategy with no selection at all matches the rule-based
  policy and beats the learned selector". Cross-fitted, the fixed arm is the
  **worst** of the three and its interval includes zero. Strategy selection does
  help once the gate is honest; it is the *learned* selection that fails to beat
  a hand-written rule.
- The final agent's design choice - a fixed strategy - was justified by that
  claim. On cross-fitted evidence the rule-based policy would have been the
  better choice (+0.0065 against +0.0050).

### R1(g) Phase V ladder

In the table above. Rung 2 (fixed, T=3) falls to **-0.0002 [-0.0056, +0.0038]**,
rung 3 (learned, T=3) to +0.0028, rung 5-noroi to +0.0052. The published
in-sample values were +0.0020, +0.0033 and +0.0057. The ordering is preserved;
the levels drop.

### R1(3) Validation and test unchanged

Confirmed. The detectors were never fitted on validation or test, so applying
the full-out-of-fold fit to them was always legitimate; cross-fitting changes
nothing there and no validation or test number in any existing report moves.
Phase V's confirmatory test pass was **not re-run**.

---

## R2. What the gate and the loop actually do

### R2(1) Predicted class of every removed component

| run | n removed | class mix |
|---|---|---|
| Phase V test agent (test) | 89 | other 84 (94%), physio 4, liver_spleen 1 |
| final agent, OOF | 208 | other 192 (92%), liver_spleen 9, physio 7 |
| final agent, val | 9 | other 8, physio 1 |
| final agent, test | 35 | other 32 (91%), liver_spleen 2, physio 1 |
| final agent, nnU-Net f0 test | 58 | other 54 (93%), physio 4 |
| final agent, Swin test | 90 | other 84 (93%), physio 6 |

**The max-over-eight-classes gate is in practice a spurious-class gate.**
Not one removal was triggered by `under_seg`, `over_seg`, `boundary`,
`fragmentation` or `merging` on any split or backbone.

### R2(2) Is the T=10 budget or the tie-break ever binding?

| configuration | cases with > 10 gated candidates | mean gated candidates/case |
|---|---|---|
| all T=10 configurations | **2 of 738** | 0.86 |
| gate 0.95 | 1 of 738 | 0.69 |

**Almost never.** The `str(c["key"])` tie-break therefore decides the edit order
on a handful of cases at most, and T=10 binds on two. The ordering concern is
real in principle and immaterial here.

### R2(3) Corrected variant

Gate restricted to `spurious_*` at >= 0.90, queue ordered by `p_failure`
descending: **+0.0049 [+0.0014, +0.0094], 240/385, dSens -0.0006**, against the
final agent's +0.0050, 239/385, -0.0006. **Indistinguishable**, which follows
directly from R2(1): the gate was already firing on spurious classes.

Validation and test runs of the corrected variant are in `au_val_corrected` and
are reported descriptively in the JSON.

### R2(4) Pseudocode, as coded

```
PHASE V TEST AGENT (phase5_agent.py, --avpkl, no --q-gate, T=10, p>=0.90)
  diagnose every predicted component ONCE, from the candidate CSV (M0 features)
  candidates <- {(component, strategy) : max_class_prob(component) >= 0.90}
  score every candidate with the learned action-value model
  for t in 1..T:
      live <- candidates not yet used, whose component still exists,
              with predicted value > eps_hat (= 0)
      if live empty: stop                      # "no candidate"
      best <- argmax over live of (value, str(key), strategy)
      apply best; record; mark its COMPONENT used   # one action per component
      # no re-diagnosis: features are never recomputed after an edit
  return the final mask                        # no revert, no best-Q selection

FINAL AGENT (phase5_agent.py, --fixed-strategy topology_remove, --q-gate, T=10)
  identical, except:
      value(candidate) = 1 if strategy == topology_remove else -1
      after applying best:
          if Q(new) - Q(old) <= 0:  revert, mark component used, continue
  stopping: live empty, or t > T
```

Two properties worth stating in the thesis because they are easy to misread.
**Candidates are computed once**, so a component's `p_failure` is always its M0
value even after neighbouring edits. **One action per component**, so a
component needing two edits gets one. Both are deliberate (`phase5_agent.py`
docstring) and both cap what the loop can do.

---

## R3. The simple baseline

`scripts/audit_baseline.py` -> `audit_baseline_tune.json`, `audit_baseline.json`.

Threshold tuned on **M0 out-of-fold only**: the largest minimum volume whose
lesion-sensitivity cost stays within -0.01. Grid:

| min volume | Dice | tumour-free with FP | FP mL | d lesion sens |
|---|---|---|---|---|
| 0.00 (M0) | 0.6888 | 289/385 | 13.256 | - |
| **0.10 (chosen)** | 0.6891 | **289/385** | 13.225 | **-0.0098** |
| 0.20 | 0.6896 | 289/385 | 13.137 | -0.0303 |
| 0.50 | 0.6875 | 289/385 | 12.863 | -0.0905 |
| 1.00 | 0.6712 | 289/385 | 12.445 | -0.1789 |
| 3.00 | 0.6070 | 289/385 | 11.366 | -0.3132 |

Applied to all three splits, against the agents:

| configuration | split | Dice | tumour-free with FP | FP mL | FN mL | HD95 | d lesion sens |
|---|---|---|---|---|---|---|---|
| M0 | OOF | 0.6888 | 289/385 | 13.256 | 6.589 | 111.29 | - |
| baseline, fixed rule (0.5 mL + organ 0.10) | OOF | 0.6265 | **289/385** | 12.364 | 14.386 | 116.83 | **-0.1498** over |
| baseline, tuned (0.1 mL) | OOF | 0.6891 | **289/385** | 13.225 | 6.676 | 109.47 | -0.0098 |
| final agent (cross-fitted) | OOF | **0.6938** | **239/385** | 12.044 | 6.595 | 94.58 | -0.0006 |
| M0 | val | 0.7061 | 22/38 | 5.158 | 3.564 | 79.95 | - |
| baseline, fixed rule | val | 0.6335 | **22/38** | 4.810 | 9.492 | 59.65 | **-0.1328** over |
| baseline, tuned | val | 0.7061 | **22/38** | 5.144 | 3.623 | 79.89 | **-0.0111** over |
| final agent | val | 0.7065 | **20/38** | 5.086 | 3.804 | 65.11 | -0.0014 |
| M0 | test | 0.6638 | 54/90 | 9.084 | 5.270 | 120.69 | - |
| baseline, fixed rule | test | 0.6231 | **54/90** | 8.300 | 10.177 | 133.03 | **-0.1251** over |
| baseline, tuned | test | 0.6636 | **54/90** | 9.063 | 5.373 | 120.76 | **-0.0133** over |
| Phase V agent | test | **0.6705** | **48/90** | - | - | - | -0.0017 |
| final agent | test | 0.6690 | **48/90** | 8.048 | 5.288 | 98.55 | -0.0011 |

"over" marks a lesion-sensitivity cost outside the -0.01 margin the agent is
held to.

**The volume baseline clears no tumour-free patient on any split at any
threshold.** 289/385 and 54/90 are unmoved from M0 throughout, because patients
carrying false positives also carry large ones that a volume floor does not
touch. The fixed Phase I/II rule costs 0.125 to 0.150 lesion sensitivity - more
than ten times the agent's budget - for less false-positive volume than the
agent removes.

**The tuned threshold does not even hold its own safety margin out of sample.**
It was chosen as the largest volume floor with dSens >= -0.01 on out-of-fold
(-0.0098, right at the limit) and then costs **-0.0111 on validation and -0.0133
on test** - over the margin on both. This is the same failure mode Phase IIIb
documented for a val-tuned threshold, now on the out-of-fold side.

Matched-sensitivity comparison is therefore not possible in the useful
direction: there is no threshold at which the baseline both meets the -0.01
margin and clears a single patient.

**This is the strongest result in the audit in the agent's favour**, and the
thesis does not currently make it. The agent is not an expensive reimplementation
of small-component removal.

---

## R4. Experiment 1 metrics

`scripts/audit_exp1_metrics.py` -> `audit_exp1_metrics.json`, computed with
`evaluate_predictions.py`'s own functions on masks taken from the stored maps at
0.5.

See the JSON for the full table (IoU, ASSD, signed and median relative volume
error, component counts, lesion-count error) for single 505, M0 and Swin f0 on
validation and test.

**Not executed:** the Phase V agent and final agent rows. The agents write
per-case metrics rather than masks, so those rows need a mask-saving re-run
(about 40 minutes per configuration for val+test). Reported as not executed
rather than approximated.

---

## R5. Per-iteration curve

`fig_audit_per_iteration.png`, `audit_per_iteration.json`. Final agent,
cross-fitted, state after exactly k accepted edits.

| k | OOF Dice | OOF tumour-free with FP | val Dice | val tumour-free with FP |
|---|---|---|---|---|
| 0 | 0.6888 | 289/385 | 0.7061 | 22/38 |
| 10 | 0.6938 | 239/385 | 0.7065 | 20/38 |

Both endpoints match the independent evaluation exactly, which is the check that
the trace is faithful. (A first version of this figure disagreed - it probed
`k*_dice` to find the last recorded step, and Dice is NaN on tumour-free cases,
so those cases silently fell back to an earlier k. Fixed by probing the field
being plotted.)

---

## R6. Statistics

### R6(1) U_stab against U_ent, component-level AUROC

`scripts/audit_auroc_boot.py` -> `audit_auroc_boot.json`. Paired bootstrap over
**patients**, 2000 resamples.

| split | components / patients | U_stab | U_ent | difference |
|---|---|---|---|---|
| test | 1508 / 163 | 0.8057 | 0.7751 | **+0.0306 [+0.0074, +0.0492]**, p=0.010 |
| out-of-fold | 6462 / 561 | 0.7709 | 0.7269 | **+0.0439 [+0.0128, +0.0734]**, p=0.004 |
| validation | 627 / 57 | 0.8526 | 0.7605 | **+0.0921 [+0.0205, +0.1428]**, p=0.007 |

**Supports the Phase II claim** and supplies the interval it never had.

### R6(2) Sign test for the Phase V test outcome split

9 improved against 0 degraded, exact two-sided: **p = 0.003906**, as the audit
expected. (For reference: final agent on Swin test 12 vs 1, p = 0.0034; on
nnU-Net f0 test 9 vs 1, p = 0.0215.)

### R6(3) Bonferroni intervals

In R1(c).

---

## R7. Self-improvement, redone

`scripts/audit_selfimprove.py` -> `audit_selfimprove.json`. Every size trained on
a patient subsample and evaluated on **one common held-out patient set**, 20
repeats, gated population.

| experience | R(a) = E[dQ \| a] | mean d_qdice | precision % |
|---|---|---|---|
| 10% of patients | +0.03832 [+0.02023, +0.06729] | +0.03704 | 88.3 [73.8, 97.6] |
| 25% | +0.04159 [+0.02304, +0.07193] | +0.03987 | 90.2 [80.6, 98.7] |
| 50% | +0.04249 [+0.02263, +0.07036] | +0.04053 | 92.2 [84.6, 98.9] |
| 100% | +0.04128 [+0.02039, +0.07373] | +0.03950 | 95.1 [88.7, 99.0] |
| **fixed (no experience)** | **+0.04305 [+0.02098, +0.07441]** | **+0.04445** | **98.3 [94.4, 100.0]** |

**The §24 conclusion holds**: the no-experience line is at or above every point,
and all intervals overlap heavily. The corrected analysis does reveal something
the old one hid - precision rises monotonically with experience, 88.3 to 95.1 -
which the previous version could not show because it scored each size on a
different population.

**The per-size endpoints pre-registered in `final_eval_plan.md` §5 l.103-104 are
not produced here.** They require an agent run per experience size per split;
the four models are saved (`fe_av_*.pkl`) and the runs are about 75 minutes
each. Reported as not executed rather than omitted silently.

---

## R8. The pre-registered H5 comparison, never reported

`phase5_analysis_plan.md` §4 named the Phase IV guarded configuration as the
secondary comparator for H5, and no report ever gave the comparison.

| | Phase V agent | Phase IV guarded | both vs M0 |
|---|---|---|---|
| test Dice | **+0.0067 [+0.0012, +0.0164]** | +0.0046 [-0.0009, +0.0140] | |
| test tumour-free with FP | 54 -> 48 (6 cleared, p=0.031) | 54 -> 48 (6 cleared, p=0.031) | identical |
| test lesion sensitivity | -0.0017 | -0.0011 | |
| val Dice | +0.0008 [-0.0010, +0.0030] | +0.0005 [-0.0013, +0.0027] | |
| val tumour-free with FP | 22 -> 20 | 22 -> 20 | identical |

The Phase V agent's Dice interval excludes zero and the Phase IV guarded one
does not, but **they clear exactly the same patients on both splits** and their
intervals overlap almost completely. The honest reading is that the Phase V
configuration is not demonstrably better than the Phase IV guarded one on the
co-primaries; it is better on Dice point estimate only.

The T=3 comparison (rung 2 against rung 3), cross-fitted: fixed **-0.0002
[-0.0056, +0.0038]** against learned **+0.0028 [+0.0016, +0.0043]**.

---

## R9. Sources and inconsistencies

| claim | status | source |
|---|---|---|
| five CV folds' best EMA pseudo-Dice 0.736-0.808, spread 0.072 | **verified**: 0.7357, 0.7496, 0.8081, 0.7357... min 0.7357, max 0.8081, **spread 0.0724** | the five `training_log_*.txt` under `nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres/fold_{3..7}` |
| total lesion voxels 8,856,911 | **verified** | `reports/convert_v2.log` l.45; `convert_v2.json` confirms 501 tumour-positive of 1014 |
| "after the fix, flipping improves uptake in only 4 of 501" | **verified by recomputation: 4 of 501 (0.8%)** | `scripts/audit_flip_recheck.py` -> `audit_flip_recheck.json` |
| post-fix uptake medians "6.14x stored against 1.46x flipped" | **does not reproduce: 3.81 / 0.89** | same script |
| "96%" orientation claim | **verified: 305 of 317** | `suv_and_postproc.md` l.398 |

On the uptake medians: the **pre-fix** pair (2.57x / 4.30x) reproduces exactly
from `mask_flip_scope.log` (2026-10-02), so the measurement definition is not in
doubt. Re-running the same test on the corrected v2 conversion gives 3.81 / 0.89,
not 6.14 / 1.46. I verified both trees behave as expected on a 60-case sample
(v2: 4.19 / 0.98, one flip-better of 60; old: 3.50 / 4.11, 33 of 60). **The
direction and the 4-of-501 count stand; those two specific medians should be
corrected or their derivation explained.** `check_mask_flip.py` hardcodes its
output path, so the recomputation was written to a new file and the pre-fix
record was not touched.

### R9(2) The single fold-0 model's two rows

63/90 vs 60/90 tumour-free with FP, sensitivity 0.5484 vs 0.5517, FN 5.45 vs
5.37. These are **two different predictions of the same weights**: the first row
is nnU-Net's own predictor writing masks (`nnUNetv2_predict`, `test_505_prep`),
the second is the 8-flip logit-fused probability map thresholded at 0.5
(`uncertainty_m0L`, `combine: logit`). The Phase III correction note records the
same distinction for Dice (0.6421 against 0.6425). Full rows are in
`audit_exp1_metrics.json`.

### R9(3) M0 FP 9.10 against 9.084

**Both are right; 9.084 is the one to quote.** 9.10 in `phase3b_cv.md` is
rounded to two decimals in a table whose other columns are also two-decimal;
9.084 in `phase5_memory.md` is the same quantity at three decimals, computed by
`phase5_eval.py` over all 200 test cases. There is no methodological difference.

### R9(4) Swin MAE pretraining corpus

`weights/swin_small_mae_best.pth` was produced by this project's own MAE
pretraining. **Whether its corpus included AutoPET could not be established from
the artefacts on disk**: the checkpoint carries no corpus manifest, and the
pretraining logs are not under `D:\data\reports`. This matters for Phase VI -
if AutoPET was in the MAE corpus, the Swin encoder saw the test images without
labels, which is a weaker but real form of contamination. **It should be
resolved from the pretraining run records before submission**, and until it is,
Phase VI's backbone comparison carries that caveat.

---

## R10. Timeline

From file timestamps, logs and git.

| when | event | evidence |
|---|---|---|
| 2026-10-15 16:42:35 | **Phase IV guarded TEST run completes** | `agent_test_pf90_final.csv`, `agent_test_pf90.log` |
| 2026-10-15 17:16:55 | `phase4_agent.md` last written | file mtime |
| **2026-10-16 14:39:42** | **`8dc2fea` Phase V pre-registration committed** | git |
| 2026-10-16 14:45:12 | action-value model fitted | `phase5_action_value.pkl` |
| 2026-10-16 16:08 - 19:31 | out-of-fold rungs 3, 4, 5, 5-noroi | `p5_oof_rung*_final.csv` |
| 2026-10-17 12:17:00 | validation eligibility check | `p5_val_final_final.csv` |
| **2026-10-17 12:35:21** | **Phase V TEST read** | `p5_test_final_final.csv` |
| 2026-10-17 15:27:52 | `669ce9c` attribution commit | git |
| 2026-10-17 18:53:17 | `b64ed9d` final-eval freeze | git |

**Did the guarded test run precede `8dc2fea`? Yes, by about 22 hours.** The
Phase V primary test pass is unaffected - it ran 22 hours *after* the plan was
committed, and the plan's primary endpoints were fixed before it. But
`phase5_analysis_plan.md` §4 names the Phase IV guarded configuration as a
**"pre-registered secondary"** comparator, and that configuration's test numbers
already existed when the plan was written. **That comparison should not be
described as pre-registered with respect to test.**

**Was the pre-test text of the plan ever copied to `docs/results`? No.** The
copy at `docs/results/phase5_analysis_plan.md` carries the later H5/H6
relabelling (md5 `a17eff48`) and differs from the text at `8dc2fea`
(md5 `3ca0dcaf`). Added as **`docs/results/phase5_analysis_plan_8dc2fea.md`**.

---

## What the thesis should change

1. **Replace the 98% figure** in `final_eval.md` §4 with 85.3% cross-fitted, and
   say the 98% was in-sample.
2. **Withdraw the claim that reasoning matches matched-rate uncertainty on
   Dice.** Keep the false-positive argument, which is stronger cross-fitted (40
   patients cleared against 15).
3. **Downgrade H4** to directionally supported, magnitude not established under
   correction.
4. **Withdraw "a fixed strategy matches or beats the learned selector"**
   (`phase5_memory.md` §6a). Cross-fitted the fixed arm is the worst of three.
   H5's verdict is unchanged but its justification must be rewritten, and the
   final agent's fixed-strategy design is no longer the best-supported choice.
5. **Correct or explain the 6.14x / 1.46x uptake medians.**
6. **Stop calling the Phase IV guarded comparison pre-registered** with respect
   to test.
7. **Add the R3 baseline comparison.** It is the strongest evidence that the
   agent is doing something a volume threshold cannot.
8. **Resolve the Swin MAE corpus question** before submission.
