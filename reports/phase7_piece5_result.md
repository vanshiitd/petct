# Piece 5 result: closed-loop active learning -- not falsified, modest effect

`scripts/phase7_active.py` -> `D:\data\reports\phase7_active.json`. Fast
proxy only (component-level mean `d_qdice`, cross-fitted over Phase IIIb
folds 3,4,5,6,7, 10 repeats per fold, piece 3's architecture and
hyperparameters held fixed -- the question here is data *order*, not model
choice). No new agent run: both curves converge by construction to piece
3's already-established 100%-data result (dDice +0.0033 OOF, 59/385 FP
cleared, `reports/phase7_piece3_result.md`), which does not beat the
hand-built policy (b).

| fraction of training data | random order, mean `d_qdice` | active order, mean `d_qdice` |
|---|---|---|
| 20% | +0.04538 | **+0.04632** |
| 40% | +0.04796 | **+0.04800** |
| 60% | +0.04828 | **+0.04849** |
| 80% | +0.04844 | **+0.04853** |
| 100% | +0.04853 | +0.04853 (identical by construction) |

Active ordering (rarest-`pred_class`-first round robin) is ahead of random
ordering at all four intermediate steps, with the gap shrinking monotonically
toward zero as both approach the shared 100%-data ceiling -- exactly the
shape expected if front-loading minority classes helps reach a given
quality level with less data, rather than raising the ceiling itself.

## Caveat, stated plainly

The percentile ranges printed for each step (e.g. 100%: `[+0.00146,
+0.12654]` for both orderings) are dominated by cross-fold variance -- fold
5's small held-out set alone produces +0.126, fold 3's produces +0.0015,
regardless of ordering -- and are **not** a valid interval on the
active-vs-random difference itself, which would need a paired comparison
within each (fold, repeat) run. That per-run data was not retained (only the
aggregated per-step statistics were written), so the comparison reported
here is a difference of means across the same fold/repeat structure, not a
bootstrapped interval on the difference. The effect is consistent in
direction and magnitude is modest (largest gap +0.00094 at 20% data, under
2% relative) -- real by the criterion's own wording (the curve rises
measurably at every intermediate step) but not large, and not formally
tested for significance.

## Verdict

Piece 5's own criterion -- "falsified if the curve does not rise measurably
above [the baseline's] flat/ceiling behaviour" -- is **not triggered**:
active ordering is ahead at 4/4 intermediate steps. This is reason 5 from
the retrospective addressed on its own narrow terms: ordering by class
rarity does make the selector more data-efficient. It does **not** change
the asymptotic ceiling, which remains piece 3's +0.0033 -- below the
hand-built policy's +0.0065. Self-improvement over time, in the sense this
piece could test it, is real but small, and does not by itself rescue the
case for a learned component over the hand-built policy.
