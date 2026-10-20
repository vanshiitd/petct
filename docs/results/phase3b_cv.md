# Phase IIIb Part 2 — patient-grouped cross-validation, and what survives it

Branch `phase3b-cv`. Everything below uses Dataset505 folds 3–7, five patient-grouped
folds over the 738 training scans only; folds 0–2 are byte-identical to Part 1 and
Datasets 501–504 are untouched.

The question Part 1 could not answer was whether its detectors were real or an artefact
of being fitted on 76 validation scans. Part 2 refits them on 738 out-of-fold scans and
asks the same questions again. Two findings reverse, one strengthens, and the headline
result turns out not to be the refinement at all.

---

## 1. What was built

Five models trained on patient-grouped folds of the training set, 500 epochs each, MAE
weights, run sequentially. Then:

- **out-of-fold maps** for all 738 training scans — each scan predicted by the single
  fold model that did not train on its patient, 8-flip TTA, logits fused then softmaxed
  once (`combine: logit`);
- **val and test maps** from a real 5-member ensemble over all 738 scans;
- candidates at `P > 0.5` (`pred`, labelled TP/FP) and at `P > 0.1` not touching
  dilated M0 (`recover`, labelled recoverable/spurious), minimum 5 voxels.

Out-of-fold is 6462 `pred` and 1086 `recover` candidates over 738 scans / 630 patients —
roughly ten times Part 1's fitting set.

### Two leaks found and fixed

**Grouping by scan, not patient.** `load()` grouped cross-validation folds by `case`.
The training split holds 738 scans from 630 patients, so 108 scans shared a patient with
another fold. Fixed with a `patient_of()` lookup that raises rather than falling back
silently. Cross-validation AUROCs moved by ≤ 0.014 and not systematically downward. The
test set is 200 scans from 200 patients, so **no test number in either Part was
affected**.

**GBT hyperparameters chosen on val.** Part 1 hardcoded `max_depth=3, max_iter=150,
learning_rate=0.08` while fitting on val, so they had seen val. This could not be
confirmed as "tuned only on out-of-fold data" and was not. Replaced with a 144-config
grouped-CV search on the out-of-fold split alone. The contamination was real but
immaterial: the grid spans only 0.0189 AUROC and the old configuration ranks about fifth
(0.8352 against the winner's 0.8362).

---

## 2. `z_rel` was a scan-edge indicator, not an anatomical prior

Part 1's spatial feature was `z_rel`, a candidate's height as a fraction of body extent.
Three probes:

- **Field of view.** AutoPET scan coverage runs 600–1983 mm, a 3.3× spread (sd ≈ 29%),
  so the same organ sits at a different `z_rel` depending on protocol.
- **Where the signal is.** The positive rate by `z_rel` decile is flat in the middle and
  extreme at the edges — the top out-of-fold `pred` decile is 84.4% false positive
  against a 39.9% base rate. Trimming to the middle eight deciles drops `z_rel`'s solo
  test AUROC from 0.5425 to **0.4665, below chance**. For recovery the `z_rel ≈ 0` bin
  holds 0 of 109 out-of-fold and 0 of 85 test positives.
- **Ablation.** Dropping `z_rel` costs the logistic model nothing (0.7977 → 0.7982) and
  the GBT 0.009, because a linear model cannot use a U-shape and a tree can.

So `z_rel` was conflating two things. It was replaced by signed distance in mm along z to
TotalSegmentator landmarks (liver, heart, bladder, kidney centroids) plus explicit
truncation distances (`edge_body_mm`, `edge_scan_mm`). That is the plan's spatial
plausibility prior done properly, with the edge effect named for what it is.

The anatomical set **improves cross-validation and lowers test** (`pred` logistic CV
0.8011 → 0.8165, test 0.7977 → 0.7672). Selecting on test would break the one-touch rule,
so selection is on out-of-fold cross-validation alone and the test drop is reported as a
consequence. See §6 for how much of that gap is population and how much is not.

Five of 738 out-of-fold scans lost TotalSegmentator (all are the tallest scans in the
cohort, 593–653 slices against a typical ~300; nnU-Net's own export pool dies at
startup). Impact measured exactly: **41 rows, 0.55% of the fitting set, 6 anatomy
columns**, median-imputed, all other feature groups intact.

---

## 3. Experiment 3 — H2, split by task

Anatomical feature set, detectors fitted on the 738 out-of-fold scans, thresholds and
model choice by grouped cross-validation there.

| | CV (out-of-fold) | val | test |
|---|---|---|---|
| **removal (`pred`)** | | | |
| a conf | 0.7443 ± 0.0409 | 0.7231 | 0.7423 |
| b unc | 0.7645 ± 0.0250 | 0.8030 | 0.7777 |
| c reason | 0.7569 ± 0.0212 | 0.7142 | 0.6824 |
| d all | 0.8165 ± 0.0198 | 0.8280 | 0.7672 |
| d all (GBT) | 0.8362 ± 0.0278 | 0.8710 | 0.8360 |
| **recovery (`recover`)** | | | |
| a conf | 0.5166 ± 0.0689 | 0.4860 | 0.5353 |
| b unc | 0.5366 ± 0.1118 | 0.5760 | 0.6051 |
| c reason | 0.5983 ± 0.0778 | 0.7016 | 0.6534 |
| d all | 0.6132 ± 0.0846 | 0.6561 | 0.6844 |
| d all (GBT) | 0.6902 ± 0.0649 | 0.7562 | 0.7601 |

Paired bootstrap on test:

| contrast | removal | recovery |
|---|---|---|
| d all − a conf | +0.0248 [+0.0052, +0.0455] | +0.1492 [+0.0840, +0.2138] |
| d all − b unc | −0.0106 [−0.0289, +0.0103] | +0.0793 [+0.0202, +0.1426] |
| c reason − a conf | **−0.0599 [−0.0927, −0.0265]** | +0.1181 [+0.0359, +0.1998] |
| GBT − logistic | +0.0689 [+0.0504, +0.0882] | +0.0757 [+0.0308, +0.1238] |

**H2 fails on removal.** With `z_rel` present the strict contrast (reasoning on top of
*all* uncertainty) was +0.0200 [+0.0021, +0.0379]; once the scan-edge confound is removed
it is −0.0106 and straddles zero, and the replacement claim is significantly negative.
Reasoning features do not beat uncertainty at deciding which predicted component is a
false positive. **Part 1's positive removal finding is withdrawn.**

**H2 holds on recovery.** All four contrasts exclude zero. Confidence alone is at chance
(0.5353). Part 1's null here (+0.0147, straddling) was a power problem — test had 27
recoverable candidates against the ensemble's 102 — and resolves with the same sign once
the fitting set grows tenfold.

The two tasks want different information, which the permutation importances show
directly. Removal: `suv_max` 0.085, `dz_bladder_mm` 0.075, `ustab_mean` 0.075. Recovery:
`edge_body_mm` 0.188, then `suv_max` 0.037 — a sub-threshold blob's recoverability is
dominated by whether it sits at a truncation boundary.

> **Uncertainty ranks false positives; reasoning finds misses.**

---

## 4. The ensemble is the largest improvement in the project

Test, n = 200, paired bootstrap over patients. M0 throughout Part 2 is the 5-fold
ensemble; the comparator is the single Dataset505 model that Part 1 used.

| | Dice | FP mL | FN mL | lesion sens | FP lesions/pt |
|---|---|---|---|---|---|
| 5-fold ensemble | 0.6638 | 9.10 | 5.27 | 0.5495 | 3.08 |
| single Dataset505 | 0.6425 | 15.96 | 5.37 | 0.5517 | 4.28 |

- Dice **+0.0213 [+0.0064, +0.0390]**
- FP volume **−6.86 mL [−10.48, −3.81]**, −43%
- FP lesions per patient **−1.20 [−1.57, −0.88]**
- lesion sensitivity −0.0022 [−0.0179, +0.0092] — no cost
- FN volume −0.10 mL [−0.75, +0.44] — no cost

Nearly half the false-positive volume removed for nothing. No refinement variant in this
report comes close, and none of them is free.

---

## 5. Experiment 4 — no prespecified threshold transfers

Thresholds tuned on 738 out-of-fold scans are stable across the five patient-grouped
folds (logistic τ_A 0.70, range 0.70–0.75 at the Dice point; 0.55, range 0.55–0.55
conservative). That is a genuine improvement over Part 1, where they came from 76 scans.
It does not help.

**Rule R, fixed before test was read:** keep variants with ΔSens ≥ −0.01 on *both*
out-of-fold and val; among those take the lowest out-of-fold FP volume; tie-break on
out-of-fold Dice.

| detector | Rule R selects | test ΔSens | test ΔFP mL | test ΔDice |
|---|---|---|---|---|
| logistic | A+B [sensitivity] | +0.0028 [−0.0280, +0.0256] | +1.63 [−0.58, +3.96] | **−0.0212 [−0.0439, −0.0010]** |
| GBT | A+B [dice] | **−0.0234 [−0.0519, −0.0028]** | **−3.22 [−5.71, −1.03]** | −0.0137 [−0.0415, +0.0095] |

The logistic pick is safe and useless: it adds false-positive volume and significantly
lowers Dice. The GBT pick does cut false positives, but its sensitivity loss is
significant and over budget — on out-of-fold it was +0.0128 and on val +0.0041.

Rule R **rejected** `A+B+reflect [sensitivity]`, which happened to land at ΔSens exactly
0.0000 on test. That cell fails on both out-of-fold (−0.0175) and val (−0.0111).
Prespecifying the rule caught exactly the test-set cherry-pick it existed to catch.

### Better ranking, worse acting

The GBT beats the logistic by +0.069 AUROC on removal and still loses at fixed
thresholds, because it converts that advantage into confident removals that take whole
lesions with them. At the `sensitivity` point on test it cuts FP volume to 3.87 mL
against the logistic's 6.33, and loses 0.0779 of lesion sensitivity against 0.0406. Its
thresholds are also less stable across folds (conservative τ_B 0.20–0.65 against
0.45–0.60). **Ranking quality does not imply a better constrained operating point.**

### How far the constraint can be pushed, per split

Best FP reduction with lesion sensitivity within one point, tuned separately on each
split — an oracle, shown to separate "tunable" from "transferable":

| | out-of-fold | val | test |
|---|---|---|---|
| logistic | τ=(0.65, 0.15), −24% | τ=(0.80, —), −3% | τ=(0.80, 0.75), −10% |
| GBT | τ=(0.55, 0.45), **−61%** | τ=(0.65, 0.40), −12% | τ=(0.80, 0.60), −27% |

The τ that achieves −61% out-of-fold is 0.55; the τ that is safe on test is 0.80. The
information exists on every split and the operating point does not survive the move
between them.

### FROC — A and B alone are pure trades

`fig_froc.png`, lesion sensitivity against false-positive lesions per patient, sweeping
thresholds, 2 × 2 over detector and split. Measured against the M0 point:

| | best sens at ≤ M0 FP/pt | fewest FP/pt at ≥ M0 sens |
|---|---|---|
| A alone, all four panels | +0.0000 | ±0.00 |
| B alone, all four panels | +0.0000 | ±0.00 |
| A+B, logistic test | +0.0028 | 3.08 → 2.94 |
| A+B, GBT val | +0.0221 | 2.41 → 1.93 |
| A+B, GBT test | **+0.0195** | **3.08 → 2.48, −19%** |

Removal alone and recovery alone never beat M0 at matched cost — they slide along the
operating characteristic the ensemble already has. Only the combination lifts it, and the
GBT lifts it seven times further than the logistic. The lift reproduces on val with the
same sign and size, so it is not purely a test artefact, but reading a point off the
Pareto front is an oracle. **The curve moves; no prespecified threshold finds the point.**

### What removal actually removes

Test, `A remove`, by ground-truth lesion volume:

**`[sensitivity]` point** — 73 of 1798 lesions lost (4.1%), 159 FP components cut, 553 mL
of FP volume:

| size | GT lesions | lost | lost % | FP comps cut | FP mL cut |
|---|---|---|---|---|---|
| < 0.5 mL | 791 | 33 | 4.2% | 109 | 21.5 |
| 0.5–1 mL | 211 | 14 | 6.6% | 15 | 10.7 |
| 1–5 mL | 467 | 17 | 3.6% | 18 | 42.8 |
| 5–20 mL | 183 | 7 | 3.8% | 10 | 115.0 |
| > 20 mL | 146 | **2** | 1.4% | 7 | 362.8 |

**`[conservative]` point** — 174 lesions lost (9.7%), 377 FP components cut, 1156 mL:

| size | GT lesions | lost | lost % | FP comps cut | FP mL cut |
|---|---|---|---|---|---|
| < 0.5 mL | 791 | 62 | 7.8% | 229 | 49.5 |
| 0.5–1 mL | 211 | 36 | 17.1% | 47 | 32.6 |
| 1–5 mL | 467 | 55 | 11.8% | 65 | 140.3 |
| 5–20 mL | 183 | 16 | 8.7% | 24 | 260.6 |
| > 20 mL | 146 | 5 | 3.4% | 12 | 673.1 |

The losses are not confined to sub-millilitre fragments. Even at the safest point the
detector deletes **two lesions larger than 20 mL and seven between 5 and 20 mL**. A
missed 20 mL lesion is not a rounding error in a reading, and this is the plainest reason
not to deploy the removal action as it stands.

---

## 6. Two mechanisms checked and excluded

### Population: test is first-visit, single-study patients only

Out-of-fold cross-validation recomputed on single-study patients (553 scans) and first
visits only (630 scans), test never read. The gap **does not close**. Restriction lowers
cross-validation for *every* model including `a conf`, which contains no positional
feature at all (0.7443 → 0.7347), so there is a −0.010 to −0.017 baseline shift from the
smaller fitting set.

The one real closure is the anatomical removal detector: `d all` gap +0.0493 → +0.0081, a
0.0413 drop against a 0.010 baseline. So the landmark features genuinely help more on
repeat and post-treatment scans than on the population test is drawn from, and that
explains part of the §2 test drop. The GBT was already flat (+0.0002) and goes negative
under restriction.

For recovery the gap has the wrong sign for a population story — cross-validation sits
*below* test by 0.05 to 0.11 — and barely moves. That is the maps difference, not the
patients.

### Train/apply mismatch: fitted on single-model maps, applied to ensemble maps

Real, and localised. Median over `pred` candidates:

| source | p_mean | uent_mean | **ustab_max** | TP rate |
|---|---|---|---|---|
| out-of-fold (single fold model) | 0.9408 | 0.1549 | **0.1443** | 60.1% |
| val ensemble | 0.9040 | 0.2298 | **0.0771** | 75.1% |
| val single-505 | 0.9067 | 0.2250 | **0.1334** | 67.1% |
| test ensemble | 0.9041 | 0.2215 | **0.0851** | 64.7% |
| test single-505 | 0.9092 | 0.2160 | **0.1494** | 58.2% |

Flip stability on the ensemble is about **half** its single-model value — averaging five
models' logits before flipping smooths the map — and out-of-fold matches the single-model
column, not the ensemble one. A second, separate shift: out-of-fold is more confident
than both (p_mean 0.941, u_ent 0.155), a property of the 590-scan fold models.

**U_epi is undefined on out-of-fold.** The `cv/oof` maps carry
`[bbox, combine, folds, p_bar, shape, spacing, u_ent, u_stab]`; `cv/val` and `cv/test`
additionally carry `u_epi`. Epistemic uncertainty is cross-model disagreement and a
single model has none. That is why `u_epi` was excluded from the feature set from the
start, so the one feature structurally absent from the fitting data is never used at
apply time either.

Rerunning Experiment 4 against single-model predictions — same detectors, same
out-of-fold thresholds, candidates rebuilt on `uncertainty_m0L` with the anatomical
features — makes transfer **worse**:

| test, `A remove [sensitivity]` | ΔDice | FP mL | ΔSens |
|---|---|---|---|
| on ensemble, logistic | −0.0040 | 9.10 → 6.33 | −0.0406 |
| on single, logistic | +0.0099 | 15.96 → 9.70 | −0.0562 |
| on ensemble, GBT | −0.0049 | 9.10 → 3.87 | −0.0779 |
| on single, GBT | **+0.0295** | 15.96 → 3.67 | −0.0612 |

So the mismatch is not the cause. It does show something else: **refinement is worth much
more on a weaker M0**. On the single model, GBT removal raises Dice by 0.0295 and cuts FP
volume by 77%; on the ensemble the same action lowers Dice. Ensembling and learned
removal clean up the same errors, and ensembling is the one that costs no sensitivity.

---

## 7. The reflection gate, rebuilt independently

The `reflect` variant in Part 1 and in §5 scores a case with a volume-weighted mean of
**the action detector's own probabilities**, so "accept only if Q improves" reduces to
"accept only if the detector agrees with itself". It is close to tautological and should
not have been reported as a reflection result.

Q was rebuilt to be independent: it regresses the case's **Dice** rather than a
component's label, sees one whole mask per case rather than one component, takes **no
detector score as input**, and is trained on masks drawn at random (each `pred` component
kept with probability u ~ U(0.5, 1), each `recover` component added with v ~ U(0, 0.35))
so it never sees the detector's decision boundary. On out-of-fold cases Q is evaluated
with the patient-grouped fold model that did not train on that patient. An empty
prediction on a tumour-free case scores Dice 1.0, since that is the state removal exists
to produce.

18,450 (mask, Dice) pairs over 738 cases. Patient-grouped cross-validation:

- R² **0.6305**, Spearman **0.7446**
- tumour-positive cases R² 0.4778, tumour-free R² 0.6158
- **within-case mask ranking median Spearman 0.6966** over 482 cases — the quantity the
  gate actually depends on

The independent gate is markedly safer than the circular one:

| variant | circular reflect (oof / val ΔSens) | Q gate (oof / val ΔSens) |
|---|---|---|
| logistic, dice point | −0.0211 / −0.0124 | **−0.0091 / −0.0083** |
| logistic, sensitivity point | −0.0175 / −0.0111 | **−0.0054 / −0.0014** |

Both Q-gated variants now pass Rule R's constraint where the circular ones failed. Rule R
still does not select them, because the gate buys safety by declining edits and so leaves
more false-positive volume than the ungated `A+B` (out-of-fold 11.36 mL against 10.13).
The gate works; it is not what the rule is optimising for.

---

## 8. Conclusions

1. **The ensemble is the best M0 available and the largest improvement so far** — Dice
   +0.0213 [+0.0064, +0.0390], false-positive volume −43%, no measurable cost in lesion
   sensitivity or missed volume. Everything after it is smaller and none of it is free.

2. **Uncertainty ranks false positives; reasoning finds misses.** On removal, reasoning
   adds nothing over all uncertainty (−0.0106, straddles zero) and is significantly worse
   as a replacement for confidence (−0.0599). On recovery, confidence is at chance and
   all four reasoning contrasts exclude zero. Part 1's positive removal finding is
   withdrawn; its recovery null is resolved positive with ten times the data.

3. **A and B alone are pure trades; A+B lifts the FROC curve, but no prespecified
   threshold transfers.** Removal alone and recovery alone never beat M0 at matched cost
   in any of four panels. The combination lifts the curve by +0.0195 lesion sensitivity
   at equal false-positive load on test, or −19% false-positive lesions at equal
   sensitivity — but that is read off the Pareto front, and Rule R, fixed on out-of-fold
   and val, selects a variant that is either useless (logistic) or over budget (GBT). The
   oracle gap is large: −61% false-positive volume is reachable within budget on
   out-of-fold, −27% on test, at different thresholds.

4. Neither the population difference nor the single-model/ensemble train-apply mismatch
   explains the failure. Both were checked and both are excluded; the second, when
   corrected, makes transfer worse.

5. Clinically, removal at its safest prespecified point still deletes two lesions above
   20 mL and seven between 5 and 20 mL out of 1798. That, not the Dice arithmetic, is the
   reason not to ship it.

## Artefacts

Maps `D:\data\cv\{oof,val,test}` (738/76/200, 8.0 GB) and `D:\data\cv_single` (junctions).
Candidates `reports\cand_cv_{oof,val,test}_pos.csv`, `reports\candL_{val,test}_pos.csv`.
Results `reports\exp3_cv_{detect,noz,anat}.json`, `reports\exp3_r_{zrel,noz,anat}_{none,single,first}.json`,
`reports\exp4_cv_anat_{logistic,gbt}.json`, `reports\exp4_single_anat_{logistic,gbt}.json`,
`reports\exp4_report.json`, `reports\exp4_final.json`, `reports\q_model.{json,pkl}`.
Figures `reports\fig_froc.png`, `reports\fig_zrel.png`.
