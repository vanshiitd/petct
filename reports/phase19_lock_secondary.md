# Phase XIX lock: Swin and nnuf0 (generality family), before any C run

This lock is committed **before** either segmenter's arms run on C. Plan
§6: arms HB and P19 only. Endpoints S19 and B19 per segmenter, with Holm
correction over the two. Each arm runs once.

## Known before the lock (A and B only)

| segmenter | fence (A) | rows A/B/C | P19 on B | B: n added, summed benefit, k | NAIVE on B: k, lesions lost | C: HB acted / P19 added |
|---|---|---|---|---|---|---|
| Swin | 53 | 2151 / 2282 / 2561 | new_cal 0.98 agree+solo | 116, +0.64, 0 | 10, 14 | 36 / 111 |
| nnuf0 | 52 | 2379 / 2520 / 2700 | ens 0.96 agree+solo | 38, +0.40, 0 | 10, 16 | 102 / 39 |

The Swin map count is 377, but 6 cases have no predicted component; the
candidate builder reported 371 cases.

## SHA-256 (`D:\data\reports`)

```
08f8c3a510ab4ed276d170bbd7fbc63b5fcb426ba2ddd367087f50048ef24857  phase19_swin_psma_hb_C.npz
3f4f9d5cd7bb3fe2ddfe3d3e47dedcc637279de89c299e4a97a2efbbd960b2c5  phase19_swin_psma_p19_C.npz
eb94d0d3438fd8ca82c92a948441d1279226d8b736aa0372fa2b659647a082f1  phase19_nnuf0_psma_hb_C.npz
752c9f8b8e4e7dc8e7e0e09e42edd473612573117b19bfd549838625bcade9f7  phase19_nnuf0_psma_p19_C.npz
0a4a7a094b05de7bff4c112d68a84aa1aca70bde4560f9713e782ce59275ecda  phase19_swin_psma_probs.npz
de14547149a40ab09c8e1a502f3bc8acc492f84116680238baed13f32e214362  phase19_nnuf0_psma_probs.npz
ce676acbae23edc428fbcbbc119eff8115a32fe7044fa3752a617d63ea1b1422  phase19_swin_psma.json
203cb54f7e23e0ca7bfb2e98a5f1a80ce6aadf8be2597b4ec849eb49d1ee4e91  phase19_nnuf0_psma.json
29a5c880bdba6c7cca4fb03791822c45618b145e64ae7dbc84069ef62a13577d  cand_swin_psma_pos.csv
97df9a31016a926ae20fbd935b3a72d9517f4bd9a2b1754cc4c1ca723b137650  cand_nnuf0_psma_pos.csv
0c778c7c20069dadd64832a86c1afd8c7c3265e02d15c5e05a2ac41f882aba74  diag_swin_psma.pkl
b0b833d7ed7874e94d05255c0650d495bf3722152f425f71bca4daa30f657536  diag_nnuf0_psma.pkl
fe4f7d33e41dd8f30f3a6cc0d49328287c1bcdf560463c5f39bcfd9297351a9f  strategy_table_swin_psma.csv
fadf89959865b61d502fc8950ab7bea303a04b303e95b8127af9bfba504fca19  strategy_table_nnuf0_psma.csv
```

`C:\psma_maps\{swin,nnuf0}_C` hold hard links to the 125 C maps of each
segmenter.

## Commands (from `D:\data\reports`, each arm once)

```
for t in swin nnuf0; do for a in hb p19; do
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase5_agent.py --maps C:/psma_maps/${t}_C \
    --cand cand_${t}_psma_pos.csv --split test --T 10 --min-pfail 1e-6 \
    --policy phase4_policy_noban.json --pfail-table phase19_${t}_psma_${a}_C.npz \
    --out-prefix p19_${t}_psma_$a
done
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase13_eval.py --split psma \
    --run p19_${t}_psma_p19 --hb p19_${t}_psma_hb --tag phase19_p13_$t
  ../petct/.venv/Scripts/python.exe ../petct/scripts/phase19_eval.py --tag ${t}_psma --arms p19
done
```
