# Phase VII summary: all five pieces, and the phase-level verdict

Pre-registered in `reports/phase7_analysis_plan.md`, before any number was
computed. Five interventions, one per reason named in the retrospective,
built and evaluated in order, each cross-fitted on the Phase IIIb folds
(3,4,5,6,7), never touching validation (used as a check only) or test
(untouched throughout this phase).

| # | piece | reason addressed | case-level dDice, OOF | tumour-free FP cleared | own verdict |
|---|---|---|---|---|---|
| -- | (a) fixed `topology_remove` | -- | +0.0033 [-0.0035, +0.0099] | 230/385 | audit reference |
| -- | **(b) hand-built policy** | -- | **+0.0065 [+0.0025, +0.0118]** | 231/385 | the bar |
| -- | (c) learned action-value model | -- | +0.0052 [+0.0022, +0.0099] | 237/385 | audit reference |
| 1 | hierarchical shrinkage | 4 (hand-built near-optimal) | +0.0029 [+0.0015, +0.0048] | 50/385 | **falsified** |
| 2 | pairwise ranking | 1-2 (signal/noise) | +0.0033 [-0.0035, +0.0099] | 59/385 | not falsified by own criterion; collapses to one action |
| 3 | class-balanced weighting | 3 (action collapse) | +0.0033 [-0.0035, +0.0100] | 59/385 | **falsified in practice** (identical to piece 2) |
| 4 | joint diagnosis+selection | 6 (gate removes decision surface) | **-0.0027 [-0.0052, -0.0003]** | 24/385 | **falsified, harmful** |
| 5 | active learning | 5 (flat self-improvement) | n/a (fast proxy only; converges to piece 3's result) | n/a | not falsified; modest, does not raise the ceiling |

## Phase-level verdict: not successful

Per the pre-registered rule -- "Phase VII is reported as not successful if
no piece's final configuration beats the hand-built policy (b) on the
primary cross-fitted metric" -- **no piece beats (b)**. The closest any
learned configuration gets is pieces 2/3/5's shared ceiling of +0.0033,
about half of (b)'s +0.0065, with a confidence interval that still straddles
zero. Piece 4 is actively harmful. Piece 1, the piece expected going in to
be the "biggest lever," performed worst of the four pieces that were
actually deployed as agents.

## What the phase established, regardless of the headline verdict

1. **The hand-built policy is not "close to" a shrinkage estimate -- it
   does something a smooth statistic trades away.** Piece 1 made this
   concrete: partial pooling toward a population mean dominated by
   `recover` (do nothing) erased the hand-built policy's per-class override
   exactly where it matters (boundary, fragmentation, under-segmentation),
   collapsing FP clearance from 231/385 to 50/385.
2. **Fixing the regression/ranking signal-to-noise problem does not fix
   training-distribution imbalance.** Piece 2 improved cross-fitted Spearman
   from 0.31 to 0.40 -- a real, measured gain -- and still collapsed onto
   `topology_remove` for 606/607 gated components, because that action is
   correct for the 135x-larger `spurious_other` class and a model optimising
   aggregate ranking quality bets on the majority answer everywhere.
3. **Reweighting the loss cannot manufacture candidates that were never
   generated.** Piece 3's class-balanced sample weights measurably reduced
   concentration in the fast proxy (99.8% -> 97.9%) but changed nothing at
   the case level (identical 59/385 FP cleared) -- the imbalance is in which
   candidate actions exist per component, not in how hard the loss tries to
   rank the ones that do.
4. **The hard `p_failure >= 0.90` gate is not redundant scaffolding --
   removing it is actively harmful.** Piece 4's calibrated regression score,
   asked to do both the act-or-not and which-strategy job at once without
   the gate, produced the only confidence interval in the entire phase
   (including the audit's own three reference rows) that excludes zero on
   the harmful side, and breached the pre-registered -0.01 lesion-sensitivity
   safety budget on both OOF and validation.
5. **Data order helps a little; data quantity and architecture do not.**
   Piece 5 showed active, class-rarity-first sampling reaches a given
   fast-proxy quality level with less data than i.i.d. order, at every
   intermediate step tested -- real self-improvement over simulated time,
   on its own narrow terms -- but both orderings converge to the same
   ceiling already set by piece 3, which remains below the hand-built
   policy.

## What this means going forward

The honest reading is that, for this problem, a well-chosen per-class hard
override (the hand-built policy) is a strong baseline that none of five
reasonable architectural interventions displaced, and that the interventions
most likely to look like progress on an aggregate proxy (pieces 1 and 2's
proxy numbers, both higher than piece 3's or the hand-built policy's own
case-level result) were the ones whose real, case-level, cross-fitted result
was worst or merely tied. Every fast-proxy number in this phase was checked
against a real agent run before being reported as a result, exactly because
of this repeated divergence -- the discipline the phase was pre-registered
to enforce caught every instance of it.

No report, figure, prediction, or script from a prior phase was modified;
every number above is new, cross-fitted, and traceable to
`reports/phase7_piece{1,2,3,4,5}_result.md` and the underlying JSONs in
`D:\data\reports\phase7_*.json`. Test remains untouched throughout this
phase, as pre-registered.
