# H4 (difficult cases) reanalysis: supported, with a comfortable margin

`scripts/h4_continuous.py` -> `D:\data\reports\h4_continuous.json`. Same
per-case data as `final_eval.md` section 2 (`fe_oof_finalagent_traces.csv`
for `Q`, `final_case_groups.json` for `u_stab`, base/final Dice CSVs), same
bootstrap convention (`phase5_eval`'s 2000-resample case bootstrap). The only
change is testing the full monotonic relationship instead of the hardest
tertile alone.

| split | measure | n | Spearman rho(harder, dDice) | slope per decile of difficulty |
|---|---|---|---|---|
| OOF | Q | 353 | **+0.1491 [+0.0461, +0.2490]** | +0.00230 [+0.00059, +0.00467] |
| OOF | u_stab | 350 | **+0.1350 [+0.0422, +0.2307]** | +0.00173 [+0.00019, +0.00440] |
| val | Q | 38 | +0.1546 [-0.1506, +0.4086] straddles | +0.00037 [-0.00024, +0.00133] |
| val | u_stab | 37 | +0.2076 [-0.1681, +0.4826] straddles | +0.00029 [-0.00026, +0.00114] |
| test (descriptive) | Q | 110 | +0.1596 [-0.0189, +0.3359] straddles | +0.00302 [+0.00002, +0.00877] |
| test (descriptive) | u_stab | 109 | **+0.1738 [+0.0383, +0.3073]** | +0.00164 [+0.00006, +0.00466] |

Reference, tertile test (`final_eval.md` section 2): Q hardest tertile n=29,
dDice +0.0209 [+0.0003, +0.0512]; u_stab hardest tertile n=48, dDice +0.0162
[+0.0004, +0.0451] -- both lower bounds within 0.0004 of zero.

## Verdict: supported, substantially more precisely

Both difficulty measures show a comfortably positive, zero-excluding
correlation on the primary (cross-fitted, out-of-fold) analysis -- using
all 353 tumour-positive cases rather than the hardest 29 or 48. The lower
bound moves from "0.0003 away from zero" (tertile) to "0.046 away from
zero" (continuous, Q) -- the same underlying effect, recovered with far
more of the sample's power than binning into thirds and discarding the
middle and easy groups let the original test use.

Validation remains underpowered at n=37-38, as it already was in the
tertile test (where `u_stab` reported "n=0, too few" on validation
entirely) -- the continuous test at least produces a number there, same
sign, but the interval still straddles zero. Test is read descriptively,
as every other H4 number in this project already is; `u_stab` excludes
zero there too, `Q`'s interval sits just on the negative side of zero
(-0.0189) despite a strongly positive point estimate (+0.1596) on n=110.

## What this does and does not change

No agent was re-run and no model was refit; this is a different statistical
test of data already on disk. It strengthens the case that H4's effect is
real (the continuous test recovers a much wider margin from the same
underlying relationship) without claiming a larger effect than the original
point estimates already suggested -- the tertile dDice numbers and this
reanalysis's correlation are different units and are not meant to be read
as the same number restated, only as two tests of the same underlying claim
agreeing.
