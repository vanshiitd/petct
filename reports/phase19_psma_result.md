# Phase XIX result: certified self-improvement on PSMA (primary segmenter)

This is the locked run. Plan: `reports/phase19_psma_selfimprove_plan.md`.
Lock: `reports/phase19_lock_psma.md` (b8351d4). Each arm ran once on C (125
patients, never used for fitting or selection).

## Primary endpoints: **both met**

| endpoint | criterion | result | met |
|---|---|---|---|
| **S19** safety | harm vs HB, exact binomial, H0: R > 0.05, α 0.05 | **0 / 125 harmed**, p = 0.0016, exact 95% UB **2.4%** | **yes** |
| **B19** benefit | paired dDice vs HB, bootstrap, one-sided p < 0.025 | **+0.0068 [+0.0022, +0.0126]**, p = 0.0005 | **yes** |

**Certified self-improvement on the new domain is confirmed on held-out
patients.**

## Key secondary endpoints (P19 vs HB on C)

| endpoint | result |
|---|---|
| FP volume | **-126.7 mL/case [-185.8, -74.4]**: HB 719 mL to P19 592 mL, -18% |
| lesion hits | +0, so lesion NI is met |
| Dice NI | met |
| HD95 | 198.0 to 194.0 mm |
| tumour-free cleared | 0 vs 0 of 6, so endpoint 1 is not met (no power, as stated) |

Scans changed: 46 of 125.

## Baselines on C (same split; each ran once)

| arm | harmed / 125 | exact 95% UB | Dice (agg.) | dDice pos vs HB | dFP mL/case [95% CI] | lesion hits vs HB |
|---|---|---|---|---|---|---|
| HB (anchor) | -- | -- | 0.1037 | -- | -- | -- |
| **P19 (certified)** | **0** | **0.024** | 0.1102 | +0.0068 | **-126.7 [-185.8, -74.4]** | **+0** |
| UNCAL (certified, raw probabilities) | 1 | 0.037 | 0.1230 | +0.0202 | -214.0 [-284.4, -150.3] | -1 |
| NAIVE (best mean on B, no harm test) | **11** | 0.142 | 0.2269 | +0.1275 | -523.1 [-631.6, -420.3] | **-20** |
| PSEUDO (pseudo-label self-training) | **4** | 0.072 | 0.1029 | -0.0009 | **+20.8** [+13.0, +28.7] | +0 |

M0 (no refinement): Dice 0.1020, FP 733.9 mL/case, 2092 lesion hits.

## Risk-benefit curve

Certification and selection on B; the curve measures outcomes on C from the
measured component tables.

| δ | configuration | added (C) | summed benefit (C) | harmed (C) | rate | exact UB |
|---|---|---|---|---|---|---|
| 0.03 | rollback (not certifiable at n = 126) | 0 | 0 | 0 | -- | -- |
| **0.05** | new_cal 0.97 agree+solo | 63 | +0.79 | 0 | 0.000 | 0.024 |
| 0.10 | new_cal 0.97 solo | 242 | +2.22 | 2 | 0.016 | 0.050 |
| 0.15 | new_cal 0.92 solo | 320 | +4.29 | 7 | 0.056 | 0.103 |
| 0.20 | new_cal 0.85 solo | 412 | +4.63 | 10 | 0.080 | 0.132 |
| naive | new_cal 0.85 none | 413 | +5.63 | 10 | 0.080 | 0.132 |

At every δ, the realised harm rate on C stays below δ.

## Validity

1000 random 50/50 patient splits of B ∪ C, with detectors fixed. Selection
uses one half and is evaluated on the other.

| δ | violation rate, LTT | violation rate, NAIVE | mean realised risk, LTT / NAIVE | mean benefit, LTT / NAIVE |
|---|---|---|---|---|
| 0.05 | **0.005** | **0.985** | 0.010 / 0.092 | +0.83 / +5.95 |
| 0.10 | **0.004** | **0.350** | 0.031 / 0.092 | +3.50 / +5.95 |

LTT keeps its promise (≤ α = 0.05) on PSMA, as on FDG (Phase XVII-b).
Naive selection breaks a 5% promise in 98.5% of splits.

## Honest reading

1. **The headline holds.** Starting from a frozen agent that was safe but
   useless on PSMA (Phase XVIII), certified adaptation on 126 labelled
   patients produces a significant Dice gain and an 18% cut in FP volume on
   125 held-out patients. It harms none of them and loses no lesion.
2. **The gain is modest in absolute terms.** Dice goes from 0.104 to 0.110.
   The base segmenter is the bottleneck (Dice 0.10 zero-shot). The agent
   removes false positives; it cannot create correct segmentation.
3. **NAIVE buys much more Dice (0.23) at a clinical price.** It harms 11
   patients (8.8%), loses **20 lesion hits**, and its 5% risk promise fails
   in 98.5% of resamples. This is the trade-off the paper must show openly.
   - The certificate does not make the update *better*. It makes its risk
     *known and bounded*.
   - The curve shows that a user who accepts δ = 0.15-0.20 recovers most of
     NAIVE's measured benefit, with that risk stated in advance.
4. **PSEUDO fails.** Self-training without labels harms 4 patients,
   increases FP volume and gives no Dice gain.
5. **The calibration ablation does not favour calibration.** UNCAL, also
   certified, gave more benefit (+0.020 vs +0.007 dDice) with 1 harmed
   patient, within its guarantee. Calibration is not what makes P19 work
   here. The LTT certificate does the safety work with or without it. The
   paper should not claim calibration as a benefit contributor on this
   evidence.

## Deviations

- **Per-segmenter locks** (recorded in the lock file).
- **Filename collision.** `phase13_eval.py --tag phase19` and
  `phase19_eval.py --tag psma` both write `phase19_eval_psma.json`, so the
  second overwrote the first. The phase13 evaluation is deterministic, makes
  no decision, and was re-run unchanged with `--tag phase19_p13`, giving
  `phase19_p13_eval_psma.json`. Its printed numbers are identical to the
  first run.
- **"M0 Dice mid" subgroup flag.** `phase13_eval` uses AutoPET tertile
  bounds, so that subgroup holds a single PSMA patient. Its dDice vs HB is
  exactly 0, so this is not a harm signal.
- **FP-volume CI.** The pre-registered secondary FP-volume CI was computed
  ad hoc (case bootstrap, 5000 resamples, seed 0), because `phase19_eval.py`
  printed only the mean.

## Pending

Swin and nnuf0 (the generality family, §6). Their PSMA maps are running; each
gets its own lock before its C runs.
