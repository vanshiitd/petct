# Post-processing on the corrected labels

The earlier post-processing sweep was tuned against mirrored labels and is void.
This redoes it on v2: organ shares remeasured, thresholds re-tuned on Dataset505's
validation predictions only, then applied once to test.

## The organ masks were mirrored too

Before anything could be remeasured, the 426 TotalSegmentator masks had to be
checked against the v2 grid. All 426 failed:

| | size | spacing | origin y | direction y |
|---|---|---|---|---|
| TotalSegmentator | (400, 400, 305) | 2.036, 2.036, 3.0 | −619.088 | **+1** |
| v2 PET | (400, 400, 305) | 2.036, 2.036, 3.0 | **+193.444** | **−1** |

Both describe the same physical span (−619.09 → +193.44); they traverse it in
opposite directions. The masks are physically correct and were correct for the old
conversion (0 mismatches against the old PET), but as numpy arrays they are
mirrored in y against everything in v2 — the mirrored-mask bug again, one level up.
An organ-overlap rule built on them would have masked the wrong side of every
patient, and no shape check anywhere would have noticed: the shapes are identical.

Two scripts would have consumed them silently. `organ_lesion_share.py` and
`postprocess.py` both compared only `.shape`; both now compare size, spacing,
origin and direction, and refuse to run on a mismatch.

The masks were moved onto the v2 grid rather than regenerated. The two grids have
identical size, spacing and extent and differ only in traversal direction, so
nearest-neighbour resampling is a pure index flip: all 426 preserved every
label's voxel count exactly, asserted per case. All 426 arrays changed, confirming
all 426 were mirrored. (A true re-run on v2's CTres would need the GPU, which the
label-efficiency runs are using, and would change nothing but the interpolation
kernel the CT was built from — ≤0.5 HU, far below organ-segmentation sensitivity.)

## Which organs are safe to mask

Measured on the 150 tumour-positive training scans that have organ masks, against
v2 labels. An organ qualifies only if real lesions essentially never occur in it.

| organ | share of lesion volume | scans affected | worst scan | verdict |
|---|---|---|---|---|
| spleen | 5.758% | 24 | 95.6% | NOT SAFE |
| liver | 5.700% | 37 | 98.1% | NOT SAFE |
| heart | 2.072% | 61 | 42.4% | NOT SAFE |
| urinary_bladder | 0.287% | 8 | 43.1% | safe |
| kidney_right | 0.144% | 7 | 19.9% | safe |
| kidney_left | 0.054% | 12 | 0.7% | safe |
| brain | 0.000% | 0 | 0.0% | safe |

Same four organs as the old-label measurement, with shifted magnitudes (spleen
2.42% → 5.76%, heart 2.90% → 2.07%).

## Tuning, on validation only

Grid: min component volume {0, 0.1, 0.2, 0.5, 1, 2, 3, 5} mL × organ overlap
{1.01 (off), 0.75, 0.5, 0.25, 0.1}, organs = brain, bladder, both kidneys. Scored
with `evaluate_predictions.py`'s own component-level FP/FN definition; the
untouched row reproduces that evaluator exactly (Dice 0.6963, FP 10.21 mL,
FN 4.28 mL, 26 tumour-free with FP), which is what validates the sweep.

| min mL | Dice | FP mL | FN mL | tumour-free with FP |
|---|---|---|---|---|
| 0.0 (raw) | **0.6963** | 10.21 | 4.28 | 26/38 |
| 0.1 | 0.6946 | 10.11 | 4.48 | 22/38 |
| 0.2 | 0.6958 | 10.04 | 4.75 | 20/38 |
| 0.5 | 0.6927 | 9.83 | 5.28 | **15/38** |
| 1.0 | 0.6901 | 9.53 | 6.59 | 14/38 |
| 2.0 | 0.6766 | 9.14 | 10.36 | 13/38 |
| 5.0 | 0.6038 | 8.59 | 15.28 | 9/38 |

**No setting improves Dice.** The Dice-optimal choice is to do nothing. Two
settings were therefore fixed on validation, both declared before test was touched:

- **primary** — max val Dice: min 0.0 mL, overlap 0.10. Organ rule only.
- **secondary** — fewest tumour-free patients with any FP, subject to val Dice loss
  < 0.005: min 0.5 mL, overlap 0.10 (loss 0.0036; min 1.0 mL loses 0.0062 and was
  excluded by that rule, not after the fact).

## Test, 200 patients / 110 tumour-positive

### 505 pretrained

| metric | raw | primary | secondary |
|---|---|---|---|
| Dice | 0.6421 | **0.6434** | 0.6400 |
| median Dice | 0.7580 | 0.7580 | 0.7555 |
| HD95 mm | 128.0 | 127.9 | 146.4 |
| FP volume mL | 15.94 | 15.80 | **15.39** |
| FN volume mL | 5.45 | 5.55 | 6.52 |
| positives scoring Dice 0 | 4 | 4 | 7 |
| empty predictions | 1 | 1 | 3 |
| **tumour-free with any FP** | 63/90 | 61/90 | **42/90** |

### 505 scratch

| metric | raw | primary | secondary |
|---|---|---|---|
| Dice | 0.6179 | 0.6178 | 0.6168 |
| median Dice | 0.7420 | 0.7420 | 0.7440 |
| HD95 mm | 146.7 | 146.7 | 138.9 |
| FP volume mL | 15.40 | 15.29 | **14.87** |
| FN volume mL | 7.20 | 7.27 | 7.98 |
| positives scoring Dice 0 | 5 | 5 | 6 |
| empty predictions | 2 | 2 | 3 |
| **tumour-free with any FP** | 69/90 | 68/90 | **47/90** |

## What this says

**Post-processing does not improve Dice on correct labels, and the organ rule is
close to useless.** Across 200 test cases the organ rule dropped 11–12 components
and moved FP volume by 1%. The 503-era sweep did find a Dice gain (0.2719 → 0.2755),
but that was a model emitting noise against mirrored labels, where deleting small
blobs could only help. That result does not transfer and should not be quoted.

**The organ rule fails for a structural reason, not a tuning one.** This model's
false positives sit in liver, spleen and heart — which hold 5.70%, 5.76% and 2.07%
of real lesion volume and so cannot be masked. The organs that *are* safe (brain
0.000%, kidneys, bladder) are ones it barely fires in. Organ masking only pays
where physiological uptake and lesion risk do not coincide, and in whole-body FDG
they largely do. No threshold fixes that.

**Small-component removal is a real trade, not an improvement.** At 0.5 mL, a third
of the tumour-free patients stop carrying any spurious finding (63→42 and 69→47) for
about 0.002 Dice. The cost is not only Dice: false-negative volume rises 11–20%,
complete misses go from 1 to 3 and 2 to 3, and for the pretrained model mean HD95
worsens 14%. Whether a third fewer patients with spurious findings is worth two
more missed patients is a clinical call, not a metric one — so both settings are
kept and reported rather than one being declared the winner.

**The honest summary is that the false-positive load is not a post-processing
problem.** Raw FP volume is ~15 mL and 70% of tumour-free patients carry some. The
from-scratch control already showed initialisation does not touch it either. It is
a property of the task and the training recipe.

## Artifacts

| | |
|---|---|
| Organ shares (v2) | `reports\organ_lesion_share_v2.json`, `organ_share_v2.log` |
| Regridded organ masks | `D:\data\totalseg_ml_v2` (426) |
| Validation sweep | `reports\sweep_505_val.json`, `sweep_505_val.log` |
| Post-processed test | `D:\data\nnunet\predictions\test_{505,505s}_pp{A,B}` |
| Per-case scores | `reports\eval_test_{505,505s}_pp{A,B}.csv` |
