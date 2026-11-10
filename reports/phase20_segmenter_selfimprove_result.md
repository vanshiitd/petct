# Phase XX result: fine-tuning the segmenter on PSMA

This is the locked run. Plan: `phase20_segmenter_selfimprove_plan.md`
(2184727). Lock: `phase20_lock.md` (bd6f4e6). C input hashes were
re-verified, then each C arm ran once.

**Headline, stated plainly.** The pre-registered primary switch at δ = 0.05
**failed to certify on B** and rolled back to P19. The fine-tuned segmenter
is a very large improvement on every quality measure. It is certified at
δ = 0.10, and at that level it **held on the locked C**.

## Primary endpoints (DEP = P19 by the rollback rule)

| endpoint | result | met |
|---|---|---|
| E1, DEP vs M0 | +0.0086 [+0.0037, +0.0147], p = 0.0005 | yes (P19's existing gain) |
| E2, DEP vs P19 | +0.0000 | **no**: DEP is P19 |
| E3, harm vs P19 | 0 / 125 | yes, trivially |

By the plan's own failure criteria, **Phase XX's primary switch failed**:
- **Why it failed on B.** FT harmed 4 of 126 patients (3.2%). With 126
  patients and two candidates at α/2, δ = 0.05 needs k ≤ 1.
- **The certificate was not the binding constraint on C.** C's 3 harmed
  patients would also fail at δ = 0.05 (p = 0.124).

## All arms on C (n = 125; 119 tumour-positive, 4518 GT lesions)

| arm | Dice (tumour-pos.) | FP mL/case | FN mL/case | lesion hits | HD95 mm | tumour-free cleared | harmed vs P19 |
|---|---|---|---|---|---|---|---|
| M0 (zero-shot, first baseline) | 0.1072 | 733.9 | 56.54 | 2092 | 198.0 | 0 / 6 | -- |
| P19 (deployed) | 0.1158 | 592.4 | 56.51 | 2094 | 194.0 | 0 / 6 | -- |
| **FT** | **0.5470** | **14.5** | **22.58** | **3547** | **90.9** | 1 / 6 | 3 |
| FT+HB | 0.5486 | 14.3 | 22.88 | 3530 | | 3 / 6 | 3 |
| FT+P20 | identical to FT+HB (no agent update certified on B) | | | | | | |

Paired dDice on tumour-positive patients:

| comparison | dDice | 95% CI | p |
|---|---|---|---|
| FT vs M0 | **+0.4398** | [+0.3959, +0.4837] | 0.0005 |
| FT vs P19 | **+0.4312** | [+0.3875, +0.4754] | 0.0005 |
| FT+HB vs M0 | +0.4415 | [+0.3978, +0.4856] | 0.0005 |

What fine-tuning changes on C, against M0:
- Dice improves by more than 0.01 for 110 of 119 tumour-positive patients and
  worsens by more than 0.01 for 1.
- FP volume falls 98%.
- Lesion hits rise 70%.
- FN volume falls 60%.

## The dial on C (certified on B, measured once on C)

| δ | deployed | harmed on C | rate | exact 95% UB | dDice vs M0 |
|---|---|---|---|---|---|
| 0.05 | P19 (rollback) | 0 | 0.000 | 0.024 | +0.0086 |
| **0.10** | **FT** | **3** | **0.024** | **0.061** | **+0.4398** |
| 0.15 | FT | 3 | 0.024 | 0.061 | +0.4398 |
| 0.20 | FT | 3 | 0.024 | 0.061 | +0.4398 |

At δ = 0.10 the certificate holds on C: the exact UB of 6.1% is below 10%,
with p = 0.0010 for H0: R > 0.10. **This is a secondary (dial) result, not
the pre-registered primary.**

## The harmed patients on C (FT vs P19)

| patient | GT lesions | hits P19 to FT | Dice P19 to FT | FP mL P19 to FT |
|---|---|---|---|---|
| PSMA_575634d308f78dba | 1 | 1 to 0 | 0.011 to 0.000 | 10 to 2 |
| PSMA_652cdc5859bf667d | 2 | 2 to 0 | 0.011 to 0.000 | 94 to 0 |
| PSMA_8b9bd3dbbb1553f1 | 120 | 106 to 102 | **0.271 to 0.603** | 314 to 3 |

- **Two patients have a real loss.** Their few small lesions are missed
  entirely. P19 "hit" them only through spurious overlap at Dice 0.011.
- **The third is "harmed" only by the lesion-hit rule.** Its Dice more than
  doubles.

Clinically, the harm definition is strict, and that is how it is kept. It is
not relaxed after the fact.

## Honest reading

1. **The bottleneck was the segmenter.**
   - **Agent-level self-improvement** (Phase XIX) moved Dice by +0.007 to
     +0.009.
   - **Segmenter-level self-improvement** on the same 126 adaptation patients
     moves it by **+0.44**, from 0.107 to 0.547, about 5×. It also removes 98%
     of the false-positive volume.
2. **The certificate did its job; it was not the obstacle.**
   - It refused the switch at a 5% harm tolerance, where B's evidence (4 of
     126) could not support it.
   - It allowed the switch at 10%, and the realised harm on locked C was
     2.4% (UB 6.1%).
   - This is the risk-benefit dial working as designed. The paper must say
     that the pre-registered 5% level was not met.
3. **The agent adds almost nothing on top of a good segmenter.**
   - FT+HB is +0.0017 Dice over FT, and clears 2 more tumour-free scans.
   - No learned agent update was certifiable on B.
   - This agrees with H5 and Phase VI: repair does not substitute for a
     better base model, and it has less to repair when the base is good.
4. **C reuse.** C was read for the Phase XIX arms before Phase XX. Nothing in
   Phase XX was chosen on C: the recipe was fixed in the plan and the
   certification used B. The reuse must still be disclosed.

## Deviations

- **Tooling commit before the lock** (e6c33d0, recorded in the lock).
- **The E1/E2 p-values** use `paired_dice`'s two-sided bootstrap p, which
  is conservative for the one-sided test in the plan. The minimum is
  1/2000 = 0.0005.
- **Pseudo-dice inflation.** nnU-Net's validation Dice on A (0.42-0.58) is
  per-case and A-only. It was not used for any decision.
