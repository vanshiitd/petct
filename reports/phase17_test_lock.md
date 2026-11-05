# Phase XVII test lock

Committed before either test run is made. Each run is made once.

## Conditions (from `reports/phase17_ltt_plan.md`, section 6)

- **Production promoted:** `new_cal`, gate 0.98, guard none.
  - Out-of-fold: 48 additions, 0 patients harmed, 0 lesions lost.
  - Benefit LCB: +0.023.
- **Validation safety check passed** (`phase17_eval_val.json`):
  - dDice +0.0000;
  - lesion hits retained +0;
  - new false positives: 0 vs 0.

## Inputs (SHA-256)

```
f80cd83f420cf899efb9d1d0ead69820cc864c69a958d0b5791f8c46c2eb7209  phase17_ltt_test.npz
0ab300ced2f7ee098322e336f6617efe4860d36ec1fb2caa35dfbb557e6093bc  phase11_eag_hb_test.npz
91c3553e96c497a6100d03dee64e8ff595e0acb6940ab87dc483a74222ec4926  cand_cv_test_pos.csv
```

## Commands (from `D:\data\reports`)

Hand-built run:

```
../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps /d/data/cv/test --cand cand_cv_test_pos.csv --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json --pfail-table phase11_eag_hb_test.npz --out-prefix p17_test_hb
```

Phase XVII run:

```
../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps /d/data/cv/test --cand cand_cv_test_pos.csv --split test --T 10 --min-pfail 1e-6 --policy phase4_policy_noban.json --pfail-table phase17_ltt_test.npz --out-prefix p17_test_ltt
```

## Evaluation

```
phase13_eval.py --split test --run p17_test_ltt --hb p17_test_hb --tag phase17
```

On test, endpoint 5 is the confirmatory result. The patient-harm rate
against δ = 0.01 is reported descriptively.

Production adds only 5 components on test (99 acted against 94 hand-built).
The expected effect is therefore small, and that is stated in advance.
