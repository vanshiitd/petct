#!/usr/bin/env python3
"""Regenerate the figures whose out-of-fold numbers were in-sample.

Four NEW files, prefixed `fig_cf_`; the originals are left in place so the
thesis can show both if it wants to. Every panel prints its plotted range
against its axis limits and says whether anything is clipped, which is the check
`plot_final_ablation.py` added after the validation bars were found cut off.

  fig_cf_ablation.png     section 21 ladder, out-of-fold bars cross-fitted
  fig_cf_difficult.png    H4 groups with 95% and Bonferroni 97.5% intervals
  fig_cf_selfimprove.png  R7 corrected curve, common held-out set, 20 repeats
  fig_cf_gate.png         R1(a) per-action gate table, in-sample vs cross-fitted
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(r"D:\data\reports")
OUT = Path(r"D:\data\petct\docs\results")


def check(ax, values, name: str) -> None:
    """Print the plotted range against the axis limits."""
    v = [x for x in values if x == x]
    lo, hi = min(v), max(v)
    y0, y1 = ax.get_ylim()
    clipped = lo < y0 or hi > y1
    print(f"    {name:34s} data [{lo:+.4f}, {hi:+.4f}]  axis [{y0:+.4f}, "
          f"{y1:+.4f}]  {'*** CLIPPED ***' if clipped else 'ok'}")


def fig_ablation() -> None:
    print("  fig_cf_ablation.png")
    cf = json.load(open(R / "audit_eval_ablation.json", encoding="utf-8"))
    pub = json.load(open(R / "final_eval_tables.json", encoding="utf-8"))["section21"]
    cfmap = {e["label"]: e for e in cf}
    order = [("M0", "baseline", None),
             ("unc", "+unc\n(0.90)", "unc_cf"),
             ("uncmatch", "+unc\n(matched)", "uncmatch_cf"),
             ("reason", "+reasoning", "reason"),
             ("refine", "+refinement", "refine"),
             ("reflect", "+reflection\n(Q gate)", "reflect"),
             ("iterate", "+iteration\n(T=10)", "iterate"),
             ("learned", "+learned\nmemory", "learned"),
             ("finalagent", "FINAL\nagent", "FINALAGENT")]
    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    oof = []
    for tag, _, cftag in order:
        if cftag is None:
            oof.append(pub["oof"]["M0"]["dice"])
        else:
            e = cfmap[cftag]
            oof.append(e["dice"]["b"] + e["dice"]["delta"])
    valv = [pub["val"][t]["dice"] for t, _, _ in order]
    testv = [pub["test"][t]["dice"] for t, _, _ in order]
    ax[0].bar(x - w, oof, w, color="#2c3e50", label="out-of-fold (cross-fitted)")
    ax[0].bar(x, valv, w, color="#7f8c8d", label="val")
    ax[0].bar(x + w, testv, w, color="#c0392b", label="test (descriptive)")
    allv = oof + valv + testv
    ax[0].set_ylim(min(allv) - 0.004, max(allv) + 0.006)
    ax[0].set_ylabel("mean Dice, tumour-positive")
    ax[0].set_title("Section 21 ablation, out-of-fold cross-fitted: Dice")
    ax[0].legend(fontsize=8, frameon=False)
    check(ax[0], allv, "Dice")

    noof, nval, ntest = [], [], []
    for tag, _, cftag in order:
        noof.append(pub["oof"]["M0"]["neg_with_fp"] if cftag is None
                    else cfmap[cftag]["fp_count"]["a"])
        nval.append(pub["val"][tag]["neg_with_fp"])
        ntest.append(pub["test"][tag]["neg_with_fp"])
    tn = {"oof": pub["oof"]["M0"]["n_neg"], "val": pub["val"]["M0"]["n_neg"],
          "test": pub["test"]["M0"]["n_neg"]}
    po = [100 * a / tn["oof"] for a in noof]
    pv = [100 * a / tn["val"] for a in nval]
    pt = [100 * a / tn["test"] for a in ntest]
    ax[1].bar(x - w, po, w, color="#2c3e50")
    ax[1].bar(x, pv, w, color="#7f8c8d")
    ax[1].bar(x + w, pt, w, color="#c0392b")
    ax[1].set_ylim(0, max(po + pv + pt) * 1.12)
    ax[1].set_ylabel("% of tumour-free patients with any FP")
    ax[1].set_title("Section 21 ablation: false-positive burden")
    check(ax[1], po + pv + pt, "% tumour-free with FP")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([l for _, l, _ in order], fontsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cf_ablation.png", dpi=160)
    plt.close(fig)


def fig_difficult() -> None:
    print("  fig_cf_difficult.png")
    h = json.load(open(R / "audit_h4_attr.json", encoding="utf-8"))["h4_oof_crossfit"]
    keys = ["Q low (hardest)", "Q mid", "Q high (easiest)",
            "u_stab high (hardest)", "u_stab mid", "u_stab low (easiest)",
            "anatomically inconsistent", "metabolically inconsistent",
            "topologically inconsistent"]
    keys = [k for k in keys if k in h and not h[k].get("too_few")]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    lows, highs = [], []
    for i, k in enumerate(keys):
        s = h[k]
        hard = "bonf_lo" in s
        ax.barh(i, s["delta"], color="#27ae60" if s["lo"] > 0 else "#95a5a6",
                height=0.55)
        ax.plot([s["lo"], s["hi"]], [i, i], color="k", lw=1.6,
                solid_capstyle="butt")
        lows += [s["lo"]]
        highs += [s["hi"]]
        if hard:
            ax.plot([s["bonf_lo"], s["bonf_hi"]], [i + 0.26, i + 0.26],
                    color="#c0392b", lw=1.6, solid_capstyle="butt")
            lows += [s["bonf_lo"]]
            highs += [s["bonf_hi"]]
        ax.text(s["hi"] + 0.0025, i, f"n={s['n']}", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(lows) - 0.004, max(highs) + 0.014)
    ax.set_xlabel("dDice, final agent - baseline (out-of-fold, cross-fitted)")
    ax.set_title("Section 22 difficult cases, cross-fitted\n"
                 "black = 95%, red = Bonferroni 97.5% (two hardest groups)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    v = [h[k]["delta"] for k in keys] + lows + highs
    vv = [x for x in v if x == x]
    x0, x1 = ax.get_xlim()
    print(f"    {'dDice (x axis)':34s} data [{min(vv):+.4f}, {max(vv):+.4f}]  "
          f"axis [{x0:+.4f}, {x1:+.4f}]  "
          f"{'*** CLIPPED ***' if min(vv) < x0 or max(vv) > x1 else 'ok'}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cf_difficult.png", dpi=160)
    plt.close(fig)


def fig_selfimprove() -> None:
    print("  fig_cf_selfimprove.png")
    s = json.load(open(R / "audit_selfimprove.json", encoding="utf-8"))
    sizes = [k for k in ("0.1", "0.25", "0.5", "1.0") if k in s]
    xs = [58, 144, 289, 578][:len(sizes)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for j, (key, lo, hi, lab) in enumerate(
            (("R_mean", "R_lo", "R_hi", "R(a) = E[dQ | a]"),
             ("prec_mean", "prec_lo", "prec_hi",
              "% of chosen actions that improve d_qdice"))):
        m = [s[k][key] for k in sizes]
        l = [s[k][lo] for k in sizes]
        u = [s[k][hi] for k in sizes]
        ax[j].errorbar(xs, m, yerr=[np.array(m) - l, np.array(u) - m],
                       fmt="o-", color="#2c3e50", capsize=3, ms=5,
                       label="learned selector")
        f = s["fixed"]
        ax[j].axhline(f[key], color="#c0392b", ls="--", lw=1.6,
                      label="fixed strategy (no experience)")
        ax[j].fill_between([min(xs) - 20, max(xs) + 20], f[lo], f[hi],
                           color="#c0392b", alpha=0.10)
        ax[j].set_xlabel("out-of-fold patients of experience")
        ax[j].set_ylabel(lab)
        allv = m + l + u + [f[key], f[lo], f[hi]]
        pad = (max(allv) - min(allv)) * 0.08
        ax[j].set_ylim(min(allv) - pad, max(allv) + pad)
        ax[j].set_xlim(min(xs) - 20, max(xs) + 20)
        check(ax[j], allv, lab[:32])
    ax[0].set_title("Internal reward, common held-out set, 20 repeats")
    ax[1].set_title("Selection precision")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cf_selfimprove.png", dpi=160)
    plt.close(fig)


def fig_gate() -> None:
    print("  fig_cf_gate.png")
    g = json.load(open(R / "audit_gate_table.json", encoding="utf-8"))
    labs = [b["bin"] for b in g["insample_bins"]]
    x = np.arange(len(labs))
    w = 0.38
    # Three panels, not two: in the lesions-lost panel the <0.60 bin is ~3000
    # and flattens every bin above 0.80, which is where the argument lives, so
    # the third panel repeats those bins on their own scale.
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    di = [b.get("mean_d_dice", np.nan) for b in g["insample_bins"]]
    dc = [b.get("mean_d_dice", np.nan) for b in g["crossfit_bins"]]
    ax[0].bar(x - w / 2, di, w, color="#95a5a6", label="in-sample (as shipped)")
    ax[0].bar(x + w / 2, dc, w, color="#2c3e50", label="cross-fitted")
    ax[0].axhline(0, color="k", lw=0.8)
    allv = [v for v in di + dc if v == v]
    ax[0].set_ylim(min(allv) * 1.15, max(allv) * 2.2)
    ax[0].set_ylabel("mean dDice per action")
    ax[0].set_title("topology_remove by gate confidence")
    ax[0].legend(fontsize=8, frameon=False)
    check(ax[0], allv, "mean dDice")

    li = [b.get("lesions_lost", 0) for b in g["insample_bins"]]
    lc = [b.get("lesions_lost", 0) for b in g["crossfit_bins"]]
    ax[1].bar(x - w / 2, li, w, color="#95a5a6", label="in-sample")
    ax[1].bar(x + w / 2, lc, w, color="#c0392b", label="cross-fitted")
    ax[1].set_ylim(0, max(li + lc) * 1.14)
    ax[1].set_ylabel("lesions lost")
    ax[1].set_title("Lesions lost by gate confidence")
    ax[1].legend(fontsize=8, frameon=False)
    for i, (a, b) in enumerate(zip(li, lc)):
        ax[1].text(i - w / 2, a + max(li + lc) * 0.02, f"{int(a)}",
                   ha="center", fontsize=8)
        ax[1].text(i + w / 2, b + max(li + lc) * 0.02, f"{int(b)}",
                   ha="center", fontsize=8)
    check(ax[1], li + lc, "lesions lost")

    zi = [v for v, l in zip(li, labs) if l in ("0.80-0.90", "0.90-0.95", ">=0.95")]
    zc = [v for v, l in zip(lc, labs) if l in ("0.80-0.90", "0.90-0.95", ">=0.95")]
    zl = [l for l in labs if l in ("0.80-0.90", "0.90-0.95", ">=0.95")]
    xz = np.arange(len(zl))
    ax[2].bar(xz - w / 2, zi, w, color="#95a5a6", label="in-sample")
    ax[2].bar(xz + w / 2, zc, w, color="#c0392b", label="cross-fitted")
    ax[2].set_ylim(0, max(zi + zc) * 1.22)
    ax[2].set_ylabel("lesions lost")
    ax[2].set_title("Lesions lost, confident bins only")
    ax[2].legend(fontsize=8, frameon=False)
    for i, (a_, b_) in enumerate(zip(zi, zc)):
        ax[2].text(i - w / 2, a_ + max(zi + zc) * 0.03, f"{int(a_)}",
                   ha="center", fontsize=9)
        ax[2].text(i + w / 2, b_ + max(zi + zc) * 0.03, f"{int(b_)}",
                   ha="center", fontsize=9)
    check(ax[2], zi + zc, "lesions lost (confident bins)")

    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=8)
    ax[2].set_xticks(xz)
    ax[2].set_xticklabels(zl, fontsize=8)
    for a in ax:
        a.set_xlabel("diagnostic confidence p_failure")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cf_gate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("axis checks (data range against axis limits):")
    fig_ablation()
    fig_difficult()
    fig_selfimprove()
    fig_gate()
    print(f"\nfour figures written to {OUT}")


if __name__ == "__main__":
    main()
