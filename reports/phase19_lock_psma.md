# Phase XIX lock: primary segmenter (nnU-Net ensemble), before any C run

This file is committed **before** any arm runs on C. Each arm runs once
after it. Plan: `reports/phase19_psma_selfimprove_plan.md`.

## What is known before the lock

These come from A and B only. No C outcome has been read.

- **Fence on A:** 52 components per case.
- **Rows:** A 2343, B 2496, C 2670.
- **Certified on B (δ 0.05, α 0.05, n 126):** 40 of 88 candidates. Fixed
  sequence breaks at k >= 1.
- **Selections on B:**

  | arm | configuration | B: n added | B: summed benefit | LCB | k harmed | lesions lost |
  |---|---|---|---|---|---|---|
  | **P19** | new_cal, 0.97, agree+solo | 64 | +0.28 | +0.0020 | 0 | 0 |
  | UNCAL | new_cal, 0.98, solo (raw) | 113 | +1.23 | +0.0054 | 0 | 0 |
  | NAIVE | new_cal, 0.85, none | 398 | +6.25 | +0.0106 | **13** | **22** |
  | PSEUDO | new_cal, 0.90, none (pseudo-label detector) | -- | -- | -- | -- | -- |

  PSEUDO is fixed by the plan, with no B selection.
- **Actions on C (counts only):**

  | arm | acted | added over HB |
  |---|---|---|
  | HB | 72 | 0 |
  | P19 | 135 | 63 |
  | NAIVE | 491 | 419 |
  | PSEUDO | 205 | 133 |
  | UNCAL | 179 | 107 |

## SHA-256 (`D:\data\reports`)

```
78648e8d051121590eea18e2b8041be9e098fe6aa05cc482db42aa30c09d1fc7  phase19_psma_hb_C.npz
b95c930fb00ec17517b3cdba25f97b62f6057252d9b79c5b3ebf2b052d7b841e  phase19_psma_p19_C.npz
44d19444840d88241df7e312ee0f6d4d965116902112e0687e99238690b66c38  phase19_psma_naive_C.npz
368a1cfd35cebde3b26e205e7eb134f5d4708890a4ce59dd474a13e8e367ff67  phase19_psma_pseudo_C.npz
a2a1acf0e31781eeae03b4fc28a0c0cc1f44cd352f31a817239b2299b8a5b115  phase19_psma_uncal_C.npz
193d29ad0c79b251743198c483817f35e2bbf44cba539d613932217d546575ee  phase19_psma_probs.npz
7a26d5b795e41c5da3962bd8992a016b862f73f3d03f52375ac8d1828857663a  phase19_psma.json
e1cec0853fdd7b203c6ce12de164d8b2d272630c3b98ec0cf679e3c2cdc0a871  cand_psma_pos.csv
c125ddb63f5a6b70d13396f8e8b8ba7c4b7b582e3fecdfc54fe65d078d76634f  diag_psma.pkl
d9cc4d04761652e0ba2a1a6c065cfd28e5e90d2f111da36e54008c192e1c2a93  strategy_table_psma.csv
```

`D:\data\cv\psma_C` holds hard links to the 125 C maps in `D:\data\cv\psma`.

## Commands (from `D:\data\reports`, each arm once)

```
for a in hb p19 naive pseudo uncal; do
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps D:/data/cv/psma_C \
    --cand cand_psma_pos.csv --split test --T 10 --min-pfail 1e-6 \
    --policy phase4_policy_noban.json --pfail-table phase19_psma_${a}_C.npz --out-prefix p19_psma_$a
done
../petct/.venv/Scripts/python.exe ../petct/scripts/phase13_eval.py --split psma --run p19_psma_p19 --hb p19_psma_hb --tag phase19
../petct/.venv/Scripts/python.exe ../petct/scripts/phase19_eval.py --tag psma
```

## Deviation: locks are per segmenter

The plan describes one lock. Swin and nnuf0 PSMA maps are still being
computed, so the primary segmenter is locked and run now. Each secondary
segmenter gets its own lock file before its C runs. The primary arms do not
depend on the secondary segmenters, and nothing selected or fitted changes.
