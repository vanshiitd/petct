#!/usr/bin/env python3
"""Phase IV: read the rule-based policy of eq. 30 off the out-of-fold table.

Input is the per-component, per-strategy measurement from
`phase4_strategy_table.py`. For each failure class, strategies are ranked by
their mean Delta-Dice on the components carrying that class, with ties broken
by lower Delta-FP volume, exactly as the pre-registered plan says.

Two guards, both from the plan:

  safety   a strategy whose mean effect on that class loses more lesions than
           it gains is dropped from the class's list, however good its Dice
           looks. Dice is indifferent between a missed lesion and a spurious
           blob; the safety constraint is not.

  gaming   a strategy whose Delta-Q > 0 coincides with Delta-Dice < 0 more
           often than with Delta-Dice > 0 is removed from the library
           altogether and the removal reported. Q is the agent's only judge,
           so a strategy that can fool it is worse than no strategy.

Writes the policy as JSON and the per-failure x per-strategy heatmap.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_ORDER = ["spurious_physio", "spurious_liver_spleen", "spurious_other",
               "missed", "under_seg", "over_seg", "fragmentation", "merging",
               "boundary", "none"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=Path(r"D:\data\reports\fig_strategy_heatmap.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.table)))
    for r in rows:
        for k in ("d_dice", "d_fp_ml", "d_q", "vol_ml_before", "vol_ml_after"):
            r[k] = float(r[k])
        for k in ("lesions_lost", "lesions_gained"):
            r[k] = int(r[k])
    print(f"{len(rows)} measured actions over "
          f"{len({r['case'] for r in rows})} cases")

    strategies = sorted({r["strategy"] for r in rows})

    # ---- gaming check, over the whole library -------------------------
    print(f"\nQ as a judge, per strategy (out-of-fold):")
    print(f"  {'strategy':18s}{'n':>7s}{'dQ>0':>8s}{'honest':>9s}{'gamed':>8s}"
          f"{'agree%':>9s}  verdict")
    banned = []
    gaming = {}
    for s in strategies:
        rs = [r for r in rows if r["strategy"] == s]
        up = [r for r in rs if r["d_q"] > 0]
        honest = sum(1 for r in up if r["d_dice"] > 0)
        gamed = sum(1 for r in up if r["d_dice"] < 0)
        agree = sum(1 for r in rs if (r["d_q"] > 0) == (r["d_dice"] > 0))
        pct = 100.0 * agree / len(rs) if rs else float("nan")
        bad = gamed > honest
        gaming[s] = {"n": len(rs), "n_dq_up": len(up), "honest": honest,
                     "gamed": gamed, "agree_pct": pct, "banned": bool(bad)}
        if bad:
            banned.append(s)
        print(f"  {s:18s}{len(rs):7d}{len(up):8d}{honest:9d}{gamed:8d}"
              f"{pct:8.1f}%  {'REMOVED (games Q)' if bad else 'kept'}")
    if banned:
        print(f"\n  removed from the library: {banned}")

    # ---- per-failure x per-strategy table -----------------------------
    cells = defaultdict(list)
    for r in rows:
        for c in (r["classes"].split("|") if r["classes"] else ["none"]):
            cells[(c, r["strategy"])].append(r)

    classes = [c for c in CLASS_ORDER if any(k[0] == c for k in cells)]
    print(f"\nmean Delta-Dice by failure class and strategy "
          f"(n in brackets; blank = never applicable)")
    head = "".join(f"{s[:11]:>13s}" for s in strategies)
    print(f"  {'class':22s}{head}")
    grid = np.full((len(classes), len(strategies)), np.nan)
    for i, c in enumerate(classes):
        line = f"  {c:22s}"
        for j, s in enumerate(strategies):
            rs = cells.get((c, s), [])
            if not rs:
                line += f"{'':>13s}"
                continue
            d = float(np.mean([r["d_dice"] for r in rs]))
            grid[i, j] = d
            line += f"{d:+9.4f}({len(rs):3d})"[:13].rjust(13)
        print(line)

    # ---- the policy ----------------------------------------------------
    policy, detail = {}, {}
    print(f"\npolicy (eq. 30): failure -> ordered strategies")
    for c in classes:
        if c == "none":
            continue
        scored = []
        for s in strategies:
            if s in banned:
                continue
            rs = cells.get((c, s), [])
            if len(rs) < 10:
                continue
            lost = sum(r["lesions_lost"] for r in rs)
            gained = sum(r["lesions_gained"] for r in rs)
            if lost > gained:
                detail.setdefault(c, []).append(
                    {"strategy": s, "dropped": "loses more lesions than it gains",
                     "lost": lost, "gained": gained})
                continue
            scored.append((float(np.mean([r["d_dice"] for r in rs])),
                           -float(np.mean([r["d_fp_ml"] for r in rs])), s,
                           len(rs), lost, gained))
        scored.sort(reverse=True)
        policy[c] = [t[2] for t in scored if t[0] > 0]
        print(f"  {c:22s}{policy[c] if policy[c] else '(nothing helps)'}")
        for t in scored:
            detail.setdefault(c, []).append(
                {"strategy": t[2], "d_dice": t[0], "d_fp_ml": -t[1],
                 "n": t[3], "lesions_lost": t[4], "lesions_gained": t[5]})

    args.out.write_text(json.dumps(
        {"policy": policy, "detail": detail, "gaming": gaming,
         "banned": banned, "classes": classes, "strategies": strategies},
        indent=1))
    print(f"\nwritten to {args.out}")

    # ---- heatmap --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.05 * len(strategies) + 4,
                                    0.5 * len(classes) + 2.5))
    v = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(strategies)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(grid[i, j]) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, label="mean $\\Delta$Dice")
    ax.set_title("Strategy effect by failure class, out-of-fold\n"
                 "(each cell: the strategy applied to every component "
                 "carrying that failure)")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=140)
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
