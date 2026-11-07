# Phase XVIII lock: PSMA tables, before any agent run

Committed **before** either agent arm runs on PSMA. Each arm runs once after
this commit. Every result is reported as it comes out (`reports/phase18_psma_plan.md`).

## What has been seen before the lock (base model only, no policy outcome)

- **Segmentation.** TotalSegmentator 377/377, with 0 failures. nnU-Net
  5-fold x 8-flip ensemble 377/377 maps (`D:\data\cv\psma`, 5.19 GB).
- **Candidates.** 9828 rows from 377 cases.

  | kind | status | count |
  |---|---|---|
  | pred | TP | 5767 |
  | pred | FP | 1742 |
  | recover | recoverable | 843 |
  | recover | spurious | 1476 |

- **Tumour status by case.** 358 positive, 19 negative. The earliest study per
  patient is even more tumour-positive than the full cohort, so as stated in
  the plan, FP-clearance endpoints have very little power.
- **Policy tables.** These record actions, not outcomes:
  - route check: |P_pickled - P_refit| = 0;
  - HB acts on 189 components;
  - P17 keeps all 189 and adds 7;
  - recovery candidates: 3 in both arms;
  - 67 cases exceed the fence of 34, where P17 adds nothing;
  - 376 cases have at least one predicted component.

## SHA-256

```
08f80d7d7a0c6023e283553d86dcc39829f9944f7c52b21483a17cd8f490aaf7  phase18_psma_hb.npz
ec4718603177d6e7f4d6d4838bc0fee9ca6aeda5f7d2955777b6ba2976c11137  phase18_psma_ltt.npz
e1cec0853fdd7b203c6ce12de164d8b2d272630c3b98ec0cf679e3c2cdc0a871  cand_psma_pos.csv
120ad338d192af2b9025bda33de846fdc066d40cca712cb35532ce9cf1cbd780  cand_psma_ctx.csv
1281d7ce333744b79b8fdc557482c596e48725271fe6003cc347f05748d3a136  phase18_psma_tables.json
```

## Commands (run once each, from `D:\data\reports`)

```
../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps D:/data/cv/psma --cand cand_psma_pos.csv --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json --pfail-table phase18_psma_hb.npz  --out-prefix p18_psma_hb
../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps D:/data/cv/psma --cand cand_psma_pos.csv --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json --pfail-table phase18_psma_ltt.npz --out-prefix p18_psma_ltt
../petct/.venv/Scripts/python.exe ../petct/scripts/phase13_eval.py --split psma --run p18_psma_ltt --hb p18_psma_hb --tag phase18
../petct/.venv/Scripts/python.exe ../petct/scripts/phase18_eval.py
```

## Pre-lock code fix (recorded as a deviation; no outcome seen)

On 2026-11-07, `phase18_psma_tables.py` first stopped in
`exp3_detect.load`. That function maps each row to an FDG patient through
`splits/autopet_v1.json`, which has no PSMA cases. The fix reads the PSMA
rows with a plain CSV reader (`rows_of`). The patient groups were never used
by the table builder.

Nothing selected, fitted or thresholded changed. The route check (exact
equality with the pickled detector) passed afterwards.
