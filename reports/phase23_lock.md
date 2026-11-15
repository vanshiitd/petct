# Phase XXIII lock: before any agent run or FT scoring on the follow-ups

Plan: `reports/phase23_followup_confirmation_plan.md` (69c66f6); tooling
93344b7. Committed before `phase5_agent.py` runs on the follow-ups and
before any FT map is scored. No outcome of this phase has been computed.

## Known before the lock (no outcome)

- **Cohort.** 143 studies of 111 patients, as planned. Every pipeline step
  succeeded for every study: preprocessing 143/143, TotalSegmentator
  143/143, M0 maps 143/143, FT maps 143/143, candidates in 143/143 studies.
  **Exclusions: none** (`phase23_exclusions.json` = []).
- **Reproducibility.** Re-predicting a stored PSMA case gives the stored M0
  map to float16 precision (2 of 56,208 mask voxels flip at 0.5) and the
  stored FT mask exactly. The re-created Level-1 detectors reproduce the
  locked C tables exactly (max difference 0, HB and P19).
- **Action tables** (counts only). 3,686 predicted components and 1,165
  recovery candidates. The fence (52, from A) is exceeded by 22 studies.
  HB acts on 69 components; P19 keeps them and adds 62; 2 recovery
  candidates in both.
- Candidate statuses (TP/FP) are written by `build_candidates.py` from the
  labels, as in every earlier phase. They were not tabulated or read.

## Decision thresholds (from the plan, n = 111, no exclusions)

| | test | rejects iff |
|---|---|---|
| H1 | FT vs P19, R > 0.10, α 0.05 | k ≤ 5 harmed patients |
| H2 | P19 vs HB, R > 0.05, α 0.05 | k ≤ 1 harmed patient |

## SHA-256 (`D:\data\reports`)

```
2520a22eabfdd4ea27ad1ea603d11d4da694538d15aef2dc363959c57392203e *phase23_psmafu_hb.npz
e283d35bc07e5894a4994ff3b3a84db4a2ebc09d1c6772a06cfe65a2741325d1 *phase23_psmafu_p19.npz
4fd9f9e28954d18924dd09ef4d5a1339ae0e52402d9e86c7f96225d54ebc1f9d *cand_psmafu_pos.csv
583de3f087d7fe7d2e7ef2d4705e69765a93db2fa9691e66817d54dc6d9ff7db *cand_psmafu_ctx.csv
37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570 *phase23_exclusions.json
17a6bb3106cd228cd3fac00ef86e10be7a37ad710f8a4c9d260c1109aec5fb18 *phase23_manifest_m0.txt
12192fc74d5dc59948aac812db9e792d5661c36631c4b1a57b53095d6686ed81 *phase23_manifest_ft.txt
b95f839e071388d00f67316bc72af074a2cb3d3e7b732a700d56be2904eb7d8e *phase23_manifest_totalseg.txt
```

## Commands (from `D:\data\reports`, each once)

```
export PETCT_RAW505=D:/data/nnunet_raw_fu/Dataset505_AutoPET_MAEprep_v2 PETCT_TOTALSEG=D:/data/totalseg_fu
for a in hb p19; do
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps D:/data/cv/psma_fu     --cand cand_psmafu_pos.csv --split test --T 10 --min-pfail 1e-6     --policy phase4_policy_noban.json --pfail-table phase23_psmafu_${a}.npz --out-prefix p23_psmafu_$a
done
PYTHONPATH=D:/data/petct ../petct/.venv/Scripts/python.exe ../petct/scripts/phase23_eval.py
```
