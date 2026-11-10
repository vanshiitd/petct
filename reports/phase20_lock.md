# Phase XX lock: before any C run

This file is committed before any Phase XX arm runs on C. Each arm then runs
once. Plan: `reports/phase20_segmenter_selfimprove_plan.md` (2184727).

## Known before the lock (A and B only)

**Fine-tuning.** All 5 A folds trained for 100 epochs, exit 0. nnU-Net's
out-of-fold mean validation Dice on A per fold:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Dice | 0.489 | 0.436 | 0.582 | 0.434 | 0.422 |

**Certification on B** (`phase20_certify.json`), n = 126, α/2 = 0.025 per
candidate:

| | M0 | P19 | FT | FT+HB |
|---|---|---|---|---|
| mean Dice, tumour-positive | 0.1186 | 0.1216 | 0.5714 | 0.5699 |
| FP mL/case | | 632.3 | 4.2 | 4.0 |
| lesion hits (of 4256 GT) | | 2038 | 3306 | |
| harmed vs P19 | | | 4 | 4 |

| δ | FT p | FT+HB p | DEP |
|---|---|---|---|
| **0.05 (primary)** | 0.2398 | 0.2398 | **P19 (rollback)** |
| 0.10 | 0.0036 | 0.0036 | FT |
| 0.15 | < 1e-4 | < 1e-4 | FT |
| 0.20 | < 1e-4 | < 1e-4 | FT |

**At the primary δ = 0.05, the switch is not certified.** By the plan's
rule, DEP = P19, and this is a failure of Phase XX's primary switch.

The 4 harmed patients on B (FT vs P19):
- **PSMA_24f62f21c04402ec.** Hits drop 12 to 6; Dice 0.176 to 0.078.
- **PSMA_4ee29f1dac1a4619.** Hits drop 3 to 2; Dice 0.013 to 0.591. It counts
  as harmed by the lesion-hit rule.
- **PSMA_8effd73b76ee3191.** Hits drop 16 to 9; Dice 0.689 to 0.463.
- **PSMA_daf98725797ce51a.** Hits stay at 7; Dice 0.495 to 0.058.

**FT+P20.** `phase19_selfimprove.py --tag ft_psma` certifies no agent update
over FT+HB on B, so it rolls back. Its C table is therefore byte-identical
to FT+HB's (same hash below). The FT fence is 96 components.

## SHA-256 (`D:\data\reports`)

```
d634ff2c994b152085b7db67d0d1aa9458435610e16fa3d961e180a7dbeb62d3  phase19_ft_psma_hb_C.npz
d634ff2c994b152085b7db67d0d1aa9458435610e16fa3d961e180a7dbeb62d3  phase19_ft_psma_p19_C.npz
8488764af70dcdddc0b05ca01df3162cba60eabbe930ffbbe0e7c19220cf5b0f  phase19_ft_psma_probs.npz
f8829c06c25a94f2aa198b84d9b7e3d56a98be365b864f982ef10b80f01f1d56  phase19_ft_psma.json
d695a7ed93496d648bcd00552269c35350a4b2e50e68ba42211c6efa2089c274  phase20_certify.json
549e48a4aaa17c176a91443c9dc1e5f80eb205d147aa201a13c7030d18777e46  cand_ft_psma_pos.csv
9c201b7ed633820c648d01cd7d5d6c4189c845753495e4ead3f744ac258d5abb  diag_ft_psma.pkl
d2cccf20f632c708ac8bd6f11812cdbc4e1cf63573a051296b34fb52ff21f5ae  strategy_table_ft_psma.csv
```

`C:\psma_maps\ft_C` holds hard links to the 125 C maps in
`C:\psma_maps\ft` (5-fold ensemble).

## Commands (from `D:\data\reports`, each once)

```
for a in hb p19; do
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps C:/psma_maps/ft_C \
    --cand cand_ft_psma_pos.csv --split test --T 10 --min-pfail 1e-6 \
    --policy phase4_policy_noban.json --pfail-table phase19_ft_psma_${a}_C.npz \
    --out-prefix p20_ft_$( [ $a = hb ] && echo hb || echo p20 )
done
../petct/.venv/Scripts/python.exe ../petct/scripts/phase20_switch.py --stage eval
```

## Notes

- P19 and M0 on C are the existing `p19_psma_p19` run (Phase XIX lock
  b8351d4). They are read, not re-run.
- **Tooling fix before any C run.** `phase19_selfimprove.py --b-tables` and
  `cv_predict.py --results/--splits` were added in e6c33d0. Locked Phase XIX
  hashes were re-checked and are unchanged.
