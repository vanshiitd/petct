# H2 (removal) reanalysis: supported with the architecture actually deployed

`scripts/h2_gbt_ablation.py` -> `D:\data\reports\h2_gbt_ablation.json`.
Same data, same patient-grouped CV discipline, same `paired_bootstrap`
function as the original H2 report (`docs/results/phase3b_cv.md` section
3) -- the only change is fitting a `HistGradientBoostingClassifier` on
`b unc` as well as `d all`, so the contrast is GBT-vs-GBT instead of
logistic-vs-logistic.

| contrast | val | test |
|---|---|---|
| original: `d all` − `b unc`, **logistic** | +0.0250 [-0.0146, +0.0623] straddles 0 | **-0.0106 [-0.0289, +0.0103] straddles 0** |
| this run: `d all` − `b unc`, **GBT** | **+0.0780 [+0.0457, +0.1080]** excludes 0 | **+0.0390 [+0.0210, +0.0580]** excludes 0 |

AUROC, test: `b unc` (GBT) 0.7970, `d all` (GBT) 0.8360.

## Verdict: H2 removal is supported, with the architecture that matters

The original negative result was real for the logistic model and remains
true for it -- an additive, linear combination of reasoning features does
not clearly beat uncertainty alone at deciding whether a predicted
component is a false positive. But `scripts/phase4_diag_model.py`, the
detector actually deployed in every later phase of this project, is a GBT,
and nobody had tested reasoning's contribution within that architecture
specifically -- `exp3_detect.py` only ever fit one GBT, on the full feature
set, and contrasted it against the full-feature logistic model (`d all
(GBT)` − `d all`, +0.0689 on test), never against an uncertainty-only GBT.

With GBT on both sides, the result flips cleanly: reasoning adds +0.0390
[+0.0210, +0.0580] AUROC on test, +0.0780 [+00457, +0.1080] on validation,
both intervals comfortably clear of zero. The mechanism fits the
permutation-importance ranking already reported for `d all (GBT)`:
`suv_max` and `dz_bladder_mm` (both reasoning features) are the top two
drivers after `ustab_mean`, ahead of every other uncertainty feature -- a
signal a tree model can use through thresholds and interactions
(an SUV cutoff conditioned on distance to the bladder, say) that an
additive linear model cannot represent regardless of how much data it
gets.

## What this does and does not change

This does not touch `phase4_diag_model.py` or retrain the deployed
detector -- that model already uses the full feature set and a GBT, so
every later phase's results already benefit from whatever this
contrast demonstrates. What changes is the **hypothesis record**: H2's
"split by task, removal fails" verdict in `docs/results/final_eval.md` was
correct for the specific comparison it ran (logistic, full features vs.
uncertainty alone) but did not characterise the deployed system, and should
be read alongside this result rather than as the last word on whether
reasoning features help removal.
