# Phase X: a lesion-safety veto -- pre-registered

Written before the nested selection, the agent runs, or any validation or
test number for this configuration is computed. Branch `phase9-class-gates`
(continuing from Phase IX, whose result motivates this).

## Disclosure: how the rule's form was found

Phase IX showed that inside the deployed gate (p_failure >= 0.90) the
hand-built pick deletes a real lesion in 6.4% of actions (cross-fitted), and
that these deletions are the agent's whole lesion-sensitivity cost. The
**form** of the rule below was found by exploring those out-of-fold actions
directly:

- all 42 harmful removals are on tumour-positive patients, none on
  tumour-free ones;
- harmful removals sit deep in the body (`edge_body_mm` median 113 mm),
  beneficial ones at the body surface (median 1 mm); this raw feature
  separates them at AUROC 0.84 with no fitting, where a 26-feature GBT
  manages 0.59 cross-fitted;
- the detector's own confidence that the case already contains a correctly
  segmented lesion (some component with p_failure < 0.3) holds for 82% of
  tumour-positive cases and 17% of tumour-free ones;
- harmful deletions are small lesions; the one large interior false positive
  in the vetoed set was 25 mL.

Because the form was chosen on out-of-fold data, **out-of-fold results for
this rule are not confirmatory**, even with nested threshold selection.
Validation and test are. This is stated here so no out-of-fold number in
this phase is read as more than a consistency check.

## The rule

Veto (score -1, so the component is left untouched) a candidate action on a
component whose class is `spurious_*`, when **all** of:

1. `edge_body_mm > t` -- the component is interior, not at the body surface;
2. the case has some other component with p_failure < `c` -- the detector is
   confident there is real tumour here, so an interior "spurious" blob may be
   another lesion;
3. volume < `v` mL -- large blobs are exempt, since the measured harm is
   small lesions and the large interior false positives are the agent's
   biggest single wins.

Everything else is the hand-built policy exactly, at the 0.90 gate.

Grid: `t` in {10, 20, 30, 50} mm, `c` in {0.3, 0.5}, `v` in {1, 5, 20, inf} mL.

## Nested selection

On the four training folds of each outer fold (3,4,5,6,7), cross-fitted
p_failure throughout, choose the configuration that **maximises lesions
saved** subject to: Dice benefit given up on tumour-positive cases <= 0.001
per tumour-positive case, and **no** tumour-free clearance given up. Ties go
to fewer vetoes. Report the held-out fold's lesions saved and costs. The
production configuration is selected the same way on all out-of-fold data.

## Endpoints

Deployed through a new script, `scripts/phase10_agent.py`, which subclasses
`Phase5Agent` and changes nothing but this veto (`phase5_agent.py` is not
modified). Compared **paired, case by case** against the hand-built
policy's own run (`au_{split}_attr_policy`), and against the current final
agent (Q gate, `au_{split}_finalagent`):

1. **Safety superiority** -- lesion hits retained, (veto - hand-built), summed
   over tumour-positive cases, bootstrap over cases: lower bound > 0.
2. **Dice non-inferiority** -- paired dDice (veto - hand-built): lower 95%
   bound >= -0.0025 (about 40% of the hand-built policy's own gain).
3. **FP-clearance non-inferiority** -- tumour-free patients cleared, at most
   2 fewer than the hand-built policy.
4. **Against the final agent (secondary)** -- does the veto beat the Q gate
   on both axes at once: higher dDice and no more lesions lost.

Validation (38 tumour-positive cases; the hand-built policy loses only 2
lesions there) is a **direction check**: lesions retained >= 0 and paired
dDice point estimate >= -0.0025.

**Test** -- untouched since the audit -- is read **once**, for this single
configuration and the hand-built policy (one new hand-built test run, since
none exists; the final agent's test run already does), if and only if
criteria 1-3 hold out-of-fold and validation passes its direction check.
Test is then the confirmatory result. If any criterion fails before that
point, test stays untouched.
