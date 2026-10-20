# Phase II: uncertainty on nnU-Net (Experiment 2), and the Phase I metric gap

Branch `phase2-uncertainty` from `main` at 2917ee5. Every threshold, weight and
cut-off in this report was chosen on validation and applied unchanged to test.

---

## Part A: Experiment 1's missing metrics

`evaluate_predictions.py` now reports IoU, ASSD, volume error and connectivity
alongside what it already had. All twelve pre-existing columns were re-derived
and reproduce **byte-exactly** (0 mismatches over 76 val cases), and the new
metrics are checked in `tests/test_evaluation_metrics.py` against masks whose
answers are worked out by hand — IoU 1/3 and Dice 1/2 on cubes sharing half
their volume, ASSD 5.5 mm on separated slabs derived voxel by voxel, anisotropic
spacing scaling the result by exactly 3, lesion sensitivity 1 of 2.

Two conventions, stated because they move the numbers:

- **HD95 and ASSD measure distance to the other mask's *volume*, not its
  border.** That is what HD95 has always done here, and changing it would break
  comparability with every earlier Phase I number, so ASSD follows the same
  field rather than mixing two. Both therefore read slightly below medpy.
- **Relative volume error is reported as a median.** Its mean is +165% on test
  while the signed error is −30.5 mL, because two cases have under 1 mL of
  tumour and a 32 mL over-prediction on a 0.39 mL lesion is +8419%. The mean of
  a ratio is not a summary of this quantity.

Connected components use scipy's default **6-connectivity** (face-adjacent),
for predictions and ground truth alike.

| run | set | Dice | IoU | HD95 mm | ASSD mm | vol err mL | rel (med) | r | ρ | lesion sens | FP lesions | FP mL | FN mL | neg with FP | empty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 505 pretrained | val | **0.6963** | **0.5805** | 79.7 | 24.42 | −65.3 | −20.5% | 0.892 | 0.910 | **0.5920** | 3.59 | 10.21 | 4.28 | 26/38 | 0 |
| 505 scratch | val | 0.6809 | 0.5586 | **73.9** | **19.64** | −82.3 | −28.1% | 0.869 | 0.938 | 0.5533 | 3.64 | 9.38 | 4.13 | 27/38 | 1 |
| 505 ensemble | val | **0.6973** | 0.5791 | 80.6 | 24.53 | −73.0 | −23.2% | 0.881 | 0.897 | 0.5823 | **3.39** | **9.33** | **3.88** | **25/38** | 0 |
| 505 pretrained | test | **0.6421** | **0.5302** | **128.0** | 43.87 | **−30.5** | −5.1% | **0.884** | 0.883 | **0.5484** | 4.25 | 15.94 | **5.45** | 63/90 | **1** |
| 505 scratch | test | 0.6179 | 0.5086 | 146.7 | 50.91 | −42.5 | −2.6% | 0.809 | 0.892 | 0.5039 | 4.05 | **15.40** | 7.20 | 69/90 | 2 |
| 505 ensemble | test | 0.6337 | 0.5224 | 138.3 | **42.14** | −37.3 | −2.8% | 0.861 | 0.895 | 0.5334 | **3.80** | 15.93 | 5.66 | **62/90** | 2 |

**The 2-model × 8-flip ensemble is not an improvement.** It is marginally ahead
on val (0.6973 against 0.6963) and clearly behind on test (0.6337 against
0.6421), because averaging in the weaker from-scratch model costs more than the
flips gain. It does reduce false-positive *lesion count* on both splits
(3.80 against 4.25 on test). Sixteen forward passes per case for that is a poor
trade if segmentation quality is the goal; the ensemble's value here is that it
makes U_epi and U_stab definable at all.

**Lesion sensitivity is the number Phase I was missing.** The pretrained model
finds 54.8% of individual test lesions while scoring 0.642 Dice. Dice is carried
by the large lesions; nearly half of distinct lesions are never touched.

---

## Experiment 2: does uncertainty predict error?

Four maps per case from 2 models × 8 flips, 326 cases, 3.9 GB. The pipeline is
verified rather than assumed: a single model's 8-flip mean reproduces nnU-Net's
own output for that model at **Dice 0.9984 with identical voxel counts**
(8941 = 8941), and U_ent peaks at 0.6934 against its ln 2 bound.

### A measurement note that changed the answers twice

AUROC was first computed from 2048-bin histograms. That is unusable here: 99.2%
of body voxels fall in one bin because P_bar is ~0 almost everywhere, and
entropy and confidence trap different shares of the error mass there (9% against
21%), so two scores that are *provably the same ranking* scored 0.95 and 0.89.
Replaced by a stratified voxel sample — every error voxel, non-errors at a fixed
per-class rate — and a Mann-Whitney estimate, which is unbiased whatever the
parameterisation. The same pathology then reappeared in float16 storage, where
87% of `conf` values round to exactly 1.0 while `u_ent` never saturates. Reported
error rates undo the sampling weights; AUROC needs no such correction.

### 1. Voxel level

An error voxel is one where `P_bar > 0.5` disagrees with truth. Two regions,
because the choice changes the answer by 0.15 and quoting only one would
overstate the result.

| split | region | base rate | U_ent | U_epi | U_stab | combined (equal w) |
|---|---|---|---|---|---|---|
| val | whole body | 0.096% | 0.9948 | 0.8695 | 0.8894 | 0.9948 |
| val | within 10 mm | 11.83% | **0.7999** | 0.7569 | 0.7782 | 0.8002 |
| test | whole body | 0.094% | 0.9964 | 0.9200 | 0.9385 | 0.9964 |
| test | within 10 mm | 12.33% | **0.8434** | 0.8123 | 0.8355 | 0.8444 |

The whole-body figure is inflated by easy background and should not be quoted on
its own. In the near-tumour region, **entropy is the best single voxel-level
score and combining adds 0.001.**

### 2. Case level

Spearman, test set:

| case score | vs 1 − Dice | vs FP volume | vs HD95 |
|---|---|---|---|
| mean U_ent in prediction | 0.7433 | 0.2991 | 0.5713 |
| mean U_stab in prediction | 0.7406 | 0.3878 | **0.6373** |
| mean U_epi in prediction | 0.6898 | 0.2345 | 0.5534 |
| summed U_stab in prediction | 0.0307 | **0.7228** | 0.0293 |

Mean uncertainty tracks *quality*; summed uncertainty tracks *how much* is
wrong. Tertiles, cut-offs from val (0.01141 / 0.05311 on mean U_stab) applied
unchanged to test:

| group | n | tumour-positive | Dice | FP mL | FN mL |
|---|---|---|---|---|---|
| low | 55 | 52 | **0.8145** | **3.24** | 5.60 |
| medium | 50 | 36 | 0.6182 | 15.69 | 11.93 |
| high | 95 | 22 | **0.2317** | **23.41** | 2.40 |

Dice falls 0.81 → 0.62 → 0.23 and FP volume rises 3 → 16 → 23 mL across groups
defined without ever seeing test. The groups are unbalanced (55/50/95) precisely
because the cut-offs were not refitted. FN volume is lowest in the high group
only because 73 of its 95 cases are tumour-free and cannot have false negatives.

### 3. Component level — where the ordering reverses

For every predicted component, true positive if it touches any true lesion.
"Tumour-free only" cannot mean TP vs FP within disease-free patients (they have
no TP by construction), so the well-posed version is FP-in-a-disease-free-patient
against all TP.

| score | val AUROC | test AUROC | test, FP(neg case) vs all TP |
|---|---|---|---|
| **U_stab mean** | **0.8531** | **0.7941** | **0.8434** |
| U_ent mean | 0.7949 | 0.7832 | 0.7982 |
| U_epi mean | 0.6280 | 0.6852 | 0.6847 |
| U_stab max | 0.7567 | 0.6615 | 0.7461 |
| U_ent max | 0.5079 | 0.5104 | 0.5270 |
| component volume | 0.279 → 0.721 inv. | 0.287 → 0.713 inv. | — |
| component SUVmax | 0.270 → 0.730 inv. | 0.281 → 0.719 inv. | — |
| combined, equal weights | 0.8129 | 0.7870 | 0.8088 |
| **combined, val-fitted** | 0.8647 | **0.8050** | **0.8437** |

Val-fitted weights: **U_stab +1.64, U_epi +0.51, U_ent −0.16**. Given flip
instability, entropy earns a *negative* weight. The fitted combination loses
0.06 from val to test — what three weights fitted on 742 components should be
expected to lose — and still wins.

This is the opposite of the voxel-level ordering, and it is the level a
reasoning module acts on.

### 4. Missed lesions, by size

839 missed test lesions, median 0.10 mL — about 8 voxels — so the aggregate is
dominated by fragments. Split by size it says something quite different:

| bin | n | % of misses | missed volume | % of missed volume | median P_bar max | mean U_ent | mean U_stab | median SUVmax | % reaching P > 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| < 0.1 mL | 420 | 50% | 12.9 mL | **1.1%** | 0.0004 | 0.039 | 0.0031 | 3.19 | **5%** |
| 0.1–0.5 | 211 | 25% | 56.4 mL | 5.0% | 0.0020 | 0.078 | 0.0082 | 4.64 | 23% |
| 0.5–1 | 58 | 7% | 44.4 mL | 3.9% | 0.0565 | 0.101 | 0.0147 | 5.78 | **41%** |
| ≥ 1 mL | 150 | 18% | **1019.3 mL** | **90.0%** | 0.0039 | 0.052 | 0.0071 | 5.54 | **31%** |
| *found, for reference* | 959 | — | 21786 mL | — | **0.9990** | 0.182 | 0.0180 | 10.62 | — |

Val agrees closely (147/67/33/55 by bin; ≥1 mL carries 85.1% of missed volume
and 31% reach P > 0.1).

Half the misses by count are sub-0.1 mL fragments carrying 1.1% of missed
volume, and for those the model shows essentially nothing: 95% never reach
probability 0.1 anywhere inside the lesion. For the lesions that matter by
volume the picture is different — **31% of ≥1 mL misses and 41% of 0.5–1 mL
misses do reach P > 0.1**, so dropping the threshold from 0.5 to 0.1 would
recover roughly a third of them. The false-positive cost of doing so is not
quantified here and would have to be before anyone acted on it.

### 5. Calibration

`phase2_reliability.png`. Near the tumour, P_bar is markedly overconfident:
predictions are pushed to the extremes while observed frequencies sit near 0.45.
Temperature fitted on val is **T = 2.98**, and it cuts test ECE from 0.0865 to
0.0387. Over the whole body ECE is 0.0004 before and after and the exercise is
vacuous at a 0.09% base rate — which is why both regions are shown rather than
only the flattering one.

### Figures

| | |
|---|---|
| `phase2_reliability.png` | reliability diagram, raw and temperature-scaled, both regions |
| `phase2_uncertainty_vs_error.png` | case-level uncertainty against 1 − Dice, FP volume and HD95 |
| `phase2_example_cases.png` | coronal MIPs: Dice 0.968, Dice 0.421, and a tumour-free case with 293.9 mL of false positive where U_stab lights up on exactly the spurious findings |

---

## Part D: component feature table

`reports/components_{val,test,train}.csv`, 24 columns, one row per predicted
component and per true lesion: TP/FP/hit/missed, volume, SUVmax / SUVmean /
SUVpeak over a 1 mL sphere, mean and max of each uncertainty map, centroid,
component counts, and the TotalSegmentator organ holding most of the component.

| set | rows | TP | FP | hit | missed |
|---|---|---|---|---|---|
| val | 1465 | 484 | 258 | 421 | 302 |
| test | 3552 | 994 | 760 | 959 | 839 |
| train (50 fit-check) | 648 | 157 | 198 | 134 | 129 |

SUV comes from the original v2 `PET.nii.gz`, not the prepared channel, which is
per-scan z-scored and would make SUVmax meaningless; the recorded crop has
zoom 1.0 so the slice is exact. Organ masks were re-gridded onto v2 earlier in
this project. **Only 9 of the 50 fit-check cases have organ masks**, so that
column is empty for 41 of them rather than guessed; val and test are complete.

---

## Verdict on H1

**H1 — "uncertainty is positively correlated with actual segmentation error" —
is supported, consistently and at every level.** Case-level mean uncertainty
reaches Spearman 0.74 against 1 − Dice and summed uncertainty 0.72 against
false-positive volume; tertiles fixed on val separate test Dice 0.81 / 0.62 /
0.23; component-level AUROC for telling a false positive from a true one is
0.79–0.85; voxel-level AUROC near the tumour is 0.80–0.84. None of this required
refitting on test.

Four qualifications matter more than the headline.

**There is one confidence baseline, not two measures.** U_ent and max(P, 1−P)
are strictly monotone functions of each other — Spearman between them is exactly
1.0 — so they are the same ranking written two ways, and U_ent should be read
throughout as *the* baseline rather than as a method competing with one. The
practical question is only whether anything beats it.

**At voxel level nothing does; at component level U_stab does.** Near the
tumour, entropy leads (0.843) and combining buys 0.001. On predicted components
the order reverses: U_stab 0.794 against entropy 0.783 on test and 0.853 against
0.795 on val, with the val-fitted combination at 0.805 and entropy carrying a
negative weight. Flip instability is measuring something the probability alone
does not, and it does so at the granularity a decision is made on.

**U_epi's weak showing is provisional.** It is the variance between exactly two
members, which is a scaled squared difference rather than a variance in any
useful sense, and two models cannot separate genuine model disagreement from the
particular pair that was trained. Its component-level AUROC of 0.685 should be
read as a lower bound pending more ensemble members, not as evidence that
epistemic uncertainty is uninformative.

**Uncertainty can triage false positives; it cannot recover the misses that
matter most.** Found lesions reach P_bar 0.999 somewhere inside them; missed
ones have a median maximum of 0.0008. For sub-0.1 mL fragments — half of all
misses but 1% of missed volume — the model is simply blind, and no threshold
recovers them. For the ≥1 mL lesions carrying 90% of missed volume, about a
third do carry some signal above P = 0.1, so a minority are recoverable at an
unmeasured false-positive cost. The honest summary is that uncertainty is a
tool for suppressing spurious findings, not for finding missed disease, and
Phase III should be designed around that asymmetry.

**For Phase III, concretely:** rank predicted components by mean U_stab, or by
the val-fitted combination. On test that separates false positives from true
ones at AUROC 0.80, and 0.84 in disease-free patients — the case where a reader
has nothing else to go on and where this model puts a spurious finding in 62 of
90 patients.
