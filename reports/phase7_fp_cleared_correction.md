# Correction: the "tumour-free FP cleared" numbers for (a)/(b)/(c)

Self-caught while building Phase VIII, after all five Phase VII pieces were
already written up. Recorded here rather than silently fixed, the way every
other bug in this project and the audit before it has been handled.

## The error

`docs/results/audit_reanalysis.md`'s R1(d) table reports a column named
`negFP` for the three reference configurations: 230/385, 231/385, 237/385
for (a) fixed `topology_remove`, (b) hand-built policy, (c) learned
action-value model. The audit's own text is explicit about what that column
is: "clears 51 tumour-free patients (289 -> 238)" for one of its rows --
**`negFP` is the number of tumour-free patients who still have a false
positive after the agent runs**, out of 385, counting down from 289 (M0's
own FP-positive count). Lower is better.

`reports/phase7_analysis_plan.md` copied those three numbers into a column
labelled **"tumour-free FP cleared"** -- higher-is-better framing -- without
recomputing them. Every later piece's own "FP cleared" number (phases 1-5,
and Phase VIII) was computed correctly and independently, via
`mcnemar_fp(final, base)`'s `b - a` (`fp_n_neg` minus the remaining count),
which *is* a genuine cleared-count, higher-is-better. The result: every
piece's own number was compared against the wrong statistic for (a)/(b)/(c)
-- a remaining count masquerading as a cleared count -- for the entire
phase.

## What changes and what does not

**dDice, the primary endpoint, is unaffected.** `paired_dice` was used
consistently throughout, for every reference row and every piece, and every
pass/fail verdict in this phase was decided on dDice. No piece's verdict
changes.

**The FP-clearance narrative changes materially** for pieces 2 and 3. The
corrected comparison:

| configuration | dDice, OOF | **FP cleared (corrected)** |
|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +00099] | 59/385 |
| (b) hand-built policy | +0.0065 [+0.0025, +0.0118] | 58/385 |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 52/385 |
| (d) shrinkage (piece 1) | +0.0029 [+0.0015, +0.0048] | 50/385 |
| (e) pairwise ranking (piece 2) | +0.0033 [-0.0035, +0.0099] | **59/385** |
| (f) class-balanced ranking (piece 3) | +0.0033 [-0.0035, +0.0100] | **59/385** |
| (g) joint scorer (piece 4) | -0.0027 [-0.0052, -0.0003] | 24/385 |

Pieces 2 and 3 do not "collapse to 59/385 from 231/385" (a nearly 4x
apparent gap, as previously written) -- they **tie row (a) exactly and
mildly beat (b)'s and (c)'s true clearance** (59 against 58 and 52). Their
failure is on dDice alone, where they tie row (a), not row (b); the
FP-clearance side of the picture was never actually bad for these two
pieces.

Piece 1 and piece 4 remain real declines on FP-clearance (50 and 24 against
the corrected 58), just smaller in magnitude than the uncorrected numbers
suggested (50 against 231, 24 against 231) -- the direction of every
existing verdict is unchanged, the size of the FP-clearance gap is smaller
for all of them.

## Files corrected

`reports/phase7_analysis_plan.md` (insertion, not rewritten in place -- the
original wrong table stays visible with this correction attached directly
below it), `reports/phase7_piece1_result.md` through
`phase7_piece4_result.md`, `reports/phase7_summary.md`, and
`scripts/phase7_piece1_eval.py` through `phase7_piece4_eval.py`'s hardcoded
`REFERENCE` dictionaries -- each piece's `*_eval.json` on disk is
regenerated with the corrected reference numbers.
