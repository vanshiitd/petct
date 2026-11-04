# Phase XIII: a context-aware per-component detector under the hand-built policy -- pre-registered

Written before any Phase XIII feature, model or agent run is computed. Branch
`phase13-detector`, from `phase12-anchored`.

## Disclosure and motivation

Phases VII to XII changed the *policy* and left the detector alone. Every
one of them ended with the deficit against the hand-built policy sitting on
one or two patients the detector confidently misreads, PETCT_22d07bdc49
(out-of-fold) and PETCT_14f931f634 (validation). Those results were read
before this plan was written.

The new features below are generic image evidence a reader uses. They were
**not** designed by looking at those patients' images or at any feature value
for them. Out-of-fold and validation remain non-confirmatory. **Test is
confirmatory.**

## What changes

**Only the per-component detector (stage 1).** Policy, gate (0.90),
strategies (`phase4_policy_noban.json`), `--T 10`, the recovery detector and
`phase5_agent.py` are all unchanged. The comparison is therefore *the same
hand-built agent, with a better detector* against the hand-built agent as it
stands.

The current detector sees each component almost in isolation. The new
features add context, computed in the prepared grid from the same PET, CT,
TotalSegmentator and M0 maps (`scripts/phase13_context_features.py`).

**Local contrast** (shell = 2 to 10 mm outside the component, excluding other
M0 foreground):
- `shell_suv_mean`
- `tbr_mean` (component SUVmean / shell)
- `tbr_max` (component SUVmax / shell)

**Edge sharpness:** `edge_contrast` = (mean SUV of the inner boundary voxels
- mean SUV of the first outer ring) / SUVmax.

**Uptake texture:**
- `suv_cv` (std / mean inside)
- `frac_hot50` (fraction of voxels >= 50% of SUVmax)

**CT texture and surroundings:**
- `hu_std`
- `frac_bone_hu` (HU > 200)
- `frac_air_hu` (HU < -400)
- `shell_frac_air`
- `shell_frac_bone`

**Anatomical neighbourhood:**
- `shell_frac_physio` (shell inside brain, heart, bladder or kidneys)
- `shell_frac_liver`

**Model profile:**
- `p_frac90` (fraction of component with P > 0.9)
- `p_shell_mean`
- `uent_shell_mean`

**Left-right symmetry.** `mirror_suv_ratio` is the SUVmean at the
component's voxels mirrored about the body midline at those slices, divided
by its own SUVmean. Physiological and brown-fat uptake is often bilateral.
It is missing if the mirror falls outside the crop.

**Within-patient context:**
- `suvmax_rank_case` (fraction of the case's components with lower SUVmax)
- `suvmax_over_case_p95` (over the 95th percentile body SUV)
- `n_pred_case`
- `dist_nearest_pred_mm` (centroid distance to the nearest other component)

Missing values are imputed with the training median, as now.

## Fitting

There is no tuning:
- same eight class heads;
- same labels (`diag_oof.pkl`);
- same HistGBT parameters (`CLF`);
- features are the current set plus the 22 above.

- **Out-of-fold:** each of Phase IIIb folds 3-7 is scored by heads fitted on
  the other four. This is exactly how `audit_crossfit.npz` was made. The script
  first reproduces `audit_crossfit.npz` with the old features (max |dP| = 0)
  as a check of the route.
- **Validation and test:** heads fitted on all out-of-fold components.
- **Recovery:** the recovery rows are copied unchanged from
  `audit_crossfit.npz` (out-of-fold) and `phase11_eag_hb_{val,test}.npz`.

The table is written as a new `--pfail-table` npz
(`phase13_det_{oof,val,test}.npz`). The agent runs exactly as the hand-built
reference does: `--min-pfail 0.90 --policy phase4_policy_noban.json --T 10`.

## Endpoints

All are paired, case by case, against the hand-built run (`au_{split}_attr_policy`).

There are two superiority routes, so each is tested at a Bonferroni level
of two-sided 0.025.

1. **FP-clearance superiority.** Among tumour-free patients: more cleared,
   with an exact two-sided McNemar p < 0.025.
2. **Dice superiority.** Paired dDice > 0, with the two-sided bootstrap p
   from `phase5_eval.paired_dice` < 0.025.
3. **Dice non-inferiority.** Paired dDice: lower 95% bound >= -0.0025.
4. **Lesion non-inferiority.** Lesion hits retained, case bootstrap: lower
   95% bound >= -5.
5. **Beats the hand-built policy overall.** (1 or 2) and 3 and 4.

**Validation** is a direction check, required for test:
- tumour-free cleared >= the hand-built policy's;
- dDice >= -0.0025;
- lesion hits retained >= -1.

**Test** is read **once**, and only if 5 holds out-of-fold and validation
passes. It is read for this detector and for one hand-built test run
(`phase11_eag_hb_test.npz`). **On test, criterion 5 is the confirmatory
result.**

## Pre-specified descriptives

- Per-class AUROC and AP, old vs new detector, out-of-fold and validation.
- Brier and ECE of max-class P.
- Number of components at the 0.90 gate.
- Permutation importance of the new feature groups on the `spurious_other`
  head, out-of-fold.
- The Phase XI proxy subgroup table.

## Deviations

Any departure is recorded with its date and reason, and with what it changed.
