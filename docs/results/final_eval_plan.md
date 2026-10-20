# Final evaluation plan (frozen configurations)

Committed **before any final-evaluation test number is computed**. Branch
`final-eval`, from `phase5-memory`.

## Status of these results

**These are descriptive tables, not new claims.** The project's confirmatory
result is Phase V's single test pass (`reports/phase5_memory.md`), and that pass
is spent. Nothing here is tuned: every threshold, detector, policy and budget is
taken as already fixed in Phases III to V, and is listed below so that a reader
can confirm nothing moved.

Each table is reported on out-of-fold, validation and test. **Test columns are
labelled descriptive throughout** and support no hypothesis test except where
explicitly pre-registered in section 6 below.

M0 throughout is the Phase IIIb 5-fold ensemble (test Dice 0.6638, FP 9.08 mL).

## 1. The final agent

Per the work plan, the final agent is:

> the loop with `p_failure` gating and the Q gate, T = 10, with a **fixed
> strategy** (`topology_remove`) rather than the learned selector, since Phase V
> section 6a showed a fixed strategy matches or beats the learned selector and
> H5 was not supported.

Fixed settings, all inherited:

| setting | value | fixed in |
|---|---|---|
| base model M0 | 5-fold ensemble | Phase IIIb |
| failure detector | `phase4_diag_model.pkl`, 26 observable features | Phase IV |
| gate | `p_failure >= 0.90` | Phase IV section 10, pre-registered in Phase V |
| Q model | `q_model.pkl` | Phase IIIb |
| Q gate | accept an action only if Delta-Q > 0 | Phase IV |
| budget | T = 10 | Phase V |
| strategy | fixed `topology_remove` | Phase V section 6a |

## 2. Section 21 ablation

Seven rungs, one component added at a time, all starting from M0. Every rung is
scored through the same `CaseContext.score()` used in Phases IV and V, so no two
rows come from different metric implementations.

| # | rung | configuration |
|---|---|---|
| 1 | Baseline | M0, untouched |
| 2 | + uncertainty | fixed `topology_remove`, gated by the **uncertainty-only** detector (features `UNC` = p_mean, p_max, uent_mean, uent_max, ustab_mean, ustab_max) at P(FP) >= 0.90, T = 1, no Q gate |
| 3 | + reasoning | fixed `topology_remove`, gated by the **reasoning** detector (`phase4_diag_model`, 26 features) at `p_failure` >= 0.90, T = 1, no Q gate |
| 4 | + refinement | full strategy set via the Phase IV rule-based policy, T = 1, no Q gate |
| 5 | + reflection | rung 4 plus the Q gate, T = 1 |
| 6 | + iteration | rung 5 with T = 10 |
| 7 | + learned strategy memory | rung 6 with the Phase V learned selector in place of the rule-based policy |

**Rung 2's detector is fitted on out-of-fold data only**, grouped by patient,
with the hyperparameters already fixed in Phase IIIb. It is a re-use of the
Phase III detector definition, not a new fit against any endpoint.

Columns: Dice (tumour-positive), tumour-free patients with any FP, FP volume
(all cases), FN volume (all cases), lesion sensitivity, HD95 (tumour-positive),
and the improved / unchanged / degraded split with `|dDice| < 0.01` as unchanged.

**A no-selection arm is included at every budget**, per Phase V section 10:
rungs 2, 3 and 6 are fixed-strategy arms, so any claim that selection helps must
beat them.

## 3. Section 22 difficult cases, and H4

Baseline against the **final agent** (section 1), split by:

- **uncertainty tertile** - mean `ustab` over the case's predicted components
- **Q tertile** - `q0`, boundaries 0.130 and 0.680 taken unchanged from Phase IV
- **the three inconsistency groups** from the Phase IV diagnosis:
  anatomically inconsistent (any component with `frac_physio > 0.5` or in liver
  or spleen), metabolically inconsistent (any component with
  `suv_max_over_liver < 1.0`), topologically inconsistent (any case with
  fragmentation or merging).

All tertile boundaries are computed **on out-of-fold** and applied unchanged to
validation and test.

**H4** - gains are larger in difficult cases - is reported as a verdict:
supported only if the hardest tertile's dDice exceeds the easiest tertile's
**and** the hardest tertile's interval excludes zero, on out-of-fold, with
validation agreeing in sign. The test column is descriptive.

## 4. Section 23 failure recovery

For the final agent: improved / unchanged / degraded percentages, mean dDice and
dHD95, iterations per case, runtime per case. Descriptive only.

## 5. Section 24 self-improvement curve

Strategy-selection performance as a function of experience. Experience is
measured in **out-of-fold patients**, at 10%, 25%, 50% and 100% of the 738-scan
out-of-fold set, sampled by patient so no patient is split across sizes.

- **R(a) = E[dQ | a]**, the mean internal reward of the actions the selector
  chooses, computed on the held-out portion of the measured action table at each
  experience size.
- **The endpoints** (Dice, tumour-free with FP) for the agent built on each
  experience size, on out-of-fold and validation.
- **The fixed-strategy arm as a flat reference line**, since it uses no
  experience at all and therefore does not vary with experience size.

The curve is descriptive. It carries no test numbers.

## 6. Phase VI: SwinUNETR, and the transfer comparison

### Backbone

MONAI SwinUNETR built exactly as `petct/models.py: build_swin_unetr` with
`ARCHS["small"]`: `feature_size=24`, `depths=(2,2,2,2)`,
`num_heads=(3,6,12,24)`, `use_v2=True`, `downsample="mergingv2"`,
`norm_name="instance"`. Weights from `weights/swin_small_mae_best.pth` via
`load_foundation_weights` (prefix `encoder_decoder.`), reporting tensors loaded
and missing. No deep supervision. Dataset505, same preprocessing and plans,
fold 0, the 738/76 split, 500 epochs.

Patch size is **[192, 96, 128]**, divisible by 32 on every axis, so the Swin
window partitioning is valid without padding.

Optimiser: nnU-Net's SGD at 1e-2 first. **If the pilot is unstable, switch that
trainer to AdamW at 1e-4 and say so in the report.** Launch detached only if the
projected wall-clock is under about 24 hours; otherwise stop and report.

### Experiment 1 for the second backbone

Swin against the **single nnU-Net fold-0 model** - not the ensemble, which would
be an unmatched comparison - on validation and test, all evaluator metrics.

### Experiments 6 and 7: transfer. Pre-registered.

Generate the Swin model's 8-flip probabilities and uncertainty for validation,
test, and the 50 training cases. Then apply the **final agent completely
unchanged** - the Q model, the `p_failure` detector, every threshold and T, all
as fitted on nnU-Net out-of-fold data - to the Swin predictions.

Reported next to the same agent applied to the single nnU-Net model: dDice,
tumour-free patients cleared, lesion sensitivity, and improved / degraded counts.

**H6 - the agent transfers to a backbone it was not built on - is supported
only if**, on the Swin predictions:

1. dDice is non-inferior to zero at a margin of **-0.01**, and
2. the tumour-free-with-any-FP count does not increase, and
3. lesion sensitivity is non-inferior at a margin of **-0.01**, and
4. at least one of dDice or the tumour-free FP count improves.

This is fixed **before** the Swin test predictions are scored. The comparison
against the single nnU-Net model uses the same four criteria, so "the agent
helps less on Swin" and "the agent helps on neither" stay distinguishable.

H6 is the only hypothesis in this document with a pre-registered verdict.
Everything in sections 2 to 5 is descriptive.

## 7. Deviations

Any departure from this document is recorded in the report with the date, the
reason, and what it changed. Configurations that cannot be executed are reported
as not executed, never silently dropped.
