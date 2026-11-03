# Phases VIII-X: where the self-improving pipeline actually stands

Three pre-registered attempts after Phase VII, each built on the last
phase's failure, all falsified. Read together they say something more
useful than any one of them: **the agent is sitting on its frontier, and the
frontier is set by the detector, not by the policy.**

## 1. Strategy choice has no headroom left

Compared case by case against the hand-built policy's own run (the
comparison none of Phase VII made), every learned selector's entire deficit
is two or three patients:

- deleting a real tumour (`topology_remove` on a `merging` component, -0.62
  and -0.52 Dice), which the hand-built policy forbids;
- failing to delete one large interior false-positive blob (+0.69 and +0.38
  Dice when deleted), which the hand-built policy does.

On the other 350 tumour-positive cases they agree. The hand-built policy's
picks are already right; Phases VII-VIII were searching a space with almost
nothing in it.

## 2. The 0.90 gate is load-bearing, not conservative (Phase IX)

With cross-fitted p_failure, the hand-built pick deletes a real lesion in
6.4% of gated actions and in **24-37% of actions in every slice below the
gate**. In-sample, the [0.80, 0.90) band looked harm-free. Relaxing the gate
buys false-positive clearance with lesions at one in four. Piece 4's global
gate removal was the extreme case of this.

## 3. Safety trades against Dice at a near-fixed rate (Phase X)

The agent's lesion cost comes from the 6.4% in the gate. Two very different
mechanisms buy it back:

| mechanism | lesions saved | Dice given up per tumour-positive case |
|---|---|---|
| Q gate (current final agent) | 20 of 24 | 0.0015 |
| interior-small-lesion veto (post-hoc) | 25 of 41 | 0.0018 |

Neither is free, neither dominates, and they cost about the same per lesion.
That is what a frontier looks like: the same trade, whichever lever pulls it.

## What would actually move the frontier

All three results point upstream, at the **diagnosis model**. Its "spurious"
label is wrong about a quarter of the time just below 0.90, and wrong in the
direction that deletes tumour; inside the gate the harmful 6.4% are small
interior lesions it cannot tell from false positives. Every downstream
policy -- hand-built, learned, gated, vetoed -- inherits that error rate and
can only trade along it.

The concrete evidence that this is improvable: `edge_body_mm` alone
separates harmful from beneficial removals at AUROC 0.84, yet the detector
that already has this feature labels both as `spurious_other` at p_failure
>= 0.90. A detector trained to separate "interior small lesion" from
"interior false positive" -- with the case-level context that the case
already contains confident tumour, which the current per-component model
never sees -- is the change most likely to push the frontier out, rather
than move along it. That is a detector-retraining phase (new cross-fitted
probability tables, then every downstream evaluation re-run), not another
policy phase.
