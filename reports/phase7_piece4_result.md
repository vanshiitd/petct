# Piece 4 result: joint diagnosis+selection scorer -- falsified, harmful

Confirmatory case-level evaluation, `scripts/phase7_piece4_eval.py` ->
`D:\data\reports\phase7_piece4_eval.json`. Deployed with `--min-pfail 0.0`
(no hard gate), `eps_hat=0.0` (default), relying entirely on the calibrated
regression score to decide both whether to act and which strategy.

| policy (cross-fitted, OOF, T=10) | dDice | tumour-free FP cleared | lesion-sens delta |
|---|---|---|---|
| (a) fixed `topology_remove` | +0.0033 [-0.0035, +0.0099] | 230/385 | -- |
| **(b) hand-built policy** | **+0.0065 [+0.0025, +0.0118]** | 231/385 | -- |
| (c) learned action-value model | +0.0052 [+0.0022, +0.0099] | 237/385 | -- |
| (d) shrinkage policy (piece 1) | +0.0029 [+0.0015, +0.0048] | 50/385 | -- |
| (e) pairwise ranking (piece 2) | +0.0033 [-0.0035, +0.0099] | 59/385 | -0.0031 |
| (f) class-balanced ranking (piece 3) | +0.0033 [-0.0035, +0.0100] | 59/385 | -0.0035 |
| (g) **joint scorer (piece 4)** | **-0.0027 [-0.0052, -0.0003]** | **24/385** | **-0.0144** |

Validation: n=38, dDice **-0.0090 [-0.0241, -0.0002]**, FP cleared 0/38,
lesion-sens delta -0.0152 -- same direction, same failure.

## Verdict: falsified, and the first piece to actively harm Dice

Piece 4 is the first configuration in the whole phase (including the audit's
original three reference rows) whose confidence interval **excludes zero on
the harmful side**: -0.0027 [-0.0052, -0.0003] OOF, -0.0090 [-0.0241, -0.0002]
on validation. It also breaches its own pre-registered safety budget: lesion
sensitivity drops by 0.0144 OOF and 0.0152 on validation, both past the
-0.01 binding margin. All three of piece 4's own criteria fail: it does not
beat piece 3 on the primary metric, it does not match piece 3's safety
margin, and it does not stay inside the safety budget at all.

## Why: the fast proxy already flagged this, and the mechanism is clear

The nested-CV fast-proxy run showed precision collapsing from ~95-98%
(pieces 1-3, hard-gated) to **46.8%** overall, with per-fold act-rates
swinging from 1.0% to 14.0% of all diagnosed components -- an order of
magnitude more volatile than anything pieces 1-3 produced. That instability
carried straight through to the real agent run.

The hard `p_failure >= 0.90` gate was never "redundant scaffolding" to be
removed -- it was doing real, calibration-insensitive work. A regression
model's raw score crossing zero is a much less reliable decision boundary
than a probability model's calibrated 0.90 cutoff: `p_failure` is trained
specifically to be a calibrated probability (Phase IV's diagnosis model),
while the regression target `d_qdice` is noisy (median magnitude ~0.0002,
reasons 1-2 again) and the model's zero-crossing is not a calibrated
decision surface in the same sense. Removing the hard gate exposed every
low-p_failure component (31,932 of 34,999 training actions have
p_failure < 0.90) to a selector whose zero threshold is only weakly informed
by the data in that region, where almost nothing should be touched -- and
the agent acted on far more of them than it should have (671/738 cases vs.
604-607 for the gated pieces), eroding both Dice and lesion sensitivity.

## Carried forward to piece 5

Piece 5 (closed-loop re-fitting with active learning) is evaluated on top of
whichever configuration has done best so far at the OOF/validation stage --
piece 2/3's result (dDice +0.0033, 59/385 FP cleared), since pieces 1 and 4
are both worse and piece 4 is actively harmful. The hard `p_failure >= 0.90`
gate is kept; piece 4's result is evidence *for* keeping the two-stage
structure, not a design to build on further.
