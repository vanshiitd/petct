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
    ap.add_argument("--table", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-ban", action="store_true",
                    help="measure and report the gaming check but do not "
                         "remove the strategies that fail it. The ban exists "
                         "to protect the Q gate; with the gate disabled (the "
                         "section 21 refinement rung) there is nothing to "
                         "game, so this is the honest library there. It is "
                         "post-hoc and is labelled as such.")
    ap.add_argument("--fig", type=Path, default=Path(r"D:\data\reports\fig_strategy_heatmap.png"))
    args = ap.parse_args()

    rows = []
    for t in args.table:
        n0 = len(rows)
        rows.extend(csv.DictReader(open(t)))
        print(f"  {len(rows) - n0:6d} rows from {t.name}")
    for r in rows:
        r.setdefault("d_qdice", r.get("d_dice", "nan"))
    for r in rows:
        for k in ("d_dice", "d_qdice", "d_fp_ml", "d_q",
                  "vol_ml_before", "vol_ml_after"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["gt_positive"] = r["gt_positive"] == "True"
        for k in ("lesions_lost", "lesions_gained"):
            r[k] = int(r[k])
    print(f"{len(rows)} measured actions over "
          f"{len({r['case'] for r in rows})} cases")

    strategies = sorted({r["strategy"] for r in rows})

    # ---- gaming check, over the whole library -------------------------
    print(f"\nQ as a judge, per strategy (out-of-fold):")
    print(f"  {'strategy':18s}{'n':>7s}{'dQ>0':>8s}{'honest':>9s}{'gamed':>8s}"
          f"{'prec%':>8s}{'agree%':>8s}  verdict")
    banned = []
    gaming = {}
    for s in strategies:
        rs = [r for r in rows if r["strategy"] == s]
        # Q was trained to predict d_qdice, so that is what "does Q agree
        # with the truth it was fitted to" must be measured against. Using the
        # tumour-positive-only Dice here would score Q as wrong on every
        # tumour-free case it correctly cleaned.
        up = [r for r in rs if r["d_q"] > 0]
        honest = sum(1 for r in up if r["d_qdice"] > 0)
        gamed = sum(1 for r in up if r["d_qdice"] < 0)
        agree = sum(1 for r in rs if (r["d_q"] > 0) == (r["d_qdice"] > 0))
        pct = 100.0 * agree / len(rs) if rs else float("nan")
        bad = gamed > honest
        decided = honest + gamed
        prec = (100.0 * honest / decided) if decided else float("nan")
        gaming[s] = {"n": len(rs), "n_dq_up": len(up), "honest": honest,
                     "gamed": gamed, "agree_pct": pct,
                     "precision_pct": prec, "banned": bool(bad)}
        if bad and not args.no_ban:
            banned.append(s)
        print(f"  {s:18s}{len(rs):7d}{len(up):8d}{honest:9d}{gamed:8d}"
              f"{prec:8.1f}%{pct:8.1f}%  "
              f"{('games Q (ban lifted)' if args.no_ban else 'REMOVED (games Q)') if bad else 'kept'}")
    if args.no_ban:
        n_fail = sum(1 for g in gaming.values() if g["banned"])
        print(f"\n  --no-ban: {n_fail} strategies fail the gaming check "
              "but are KEPT (this is the post-hoc variant)")
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
            rs = [r for r in cells.get((c, s), []) if r["gt_positive"]]
            if not rs:
                line += f"{'':>13s}"
                continue
            d = float(np.nanmean([r["d_dice"] for r in rs]))
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
            allrs = cells.get((c, s), [])
            rs = [r for r in allrs if r["gt_positive"]]
            if len(allrs) < 10:
                continue
            lost = sum(r["lesions_lost"] for r in allrs)
            gained = sum(r["lesions_gained"] for r in allrs)
            if lost > gained:
                detail.setdefault(c, []).append(
                    {"strategy": s, "dropped": "loses more lesions than it gains",
                     "lost": lost, "gained": gained})
                continue
            dd = float(np.nanmean([r["d_dice"] for r in rs])) if rs else 0.0
            scored.append((dd if dd == dd else 0.0,
                           -float(np.mean([r["d_fp_ml"] for r in allrs])), s,
                           len(allrs), lost, gained))
        scored.sort(reverse=True)
        policy[c] = [t[2] for t in scored if t[0] > 0]
        print(f"  {c:22s}{policy[c] if policy[c] else '(nothing helps)'}")
        for t in scored:
            detail.setdefault(c, []).append(
                {"strategy": t[2], "d_dice": t[0], "d_fp_ml": -t[1],
                 "n": t[3], "lesions_lost": t[4], "lesions_gained": t[5]})

    args.out.write_text(json.dumps(
        {"policy": policy, "detail": detail, "gaming": gaming,
         "banned": banned, "classes": classes, "strategies": strategies,
         "variant": "no_ban_posthoc" if args.no_ban else "preregistered",
         "tables": [str(t) for t in args.table]},
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
