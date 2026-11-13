#!/usr/bin/env python3
"""Post-hoc, descriptive analyses and figures for the paper.

    D:\\data\\petct\\.venv\\Scripts\\python paper/make_posthoc.py

Reads only the per-patient tables that the locked evaluation runs wrote to
D:\\data\\reports (no model is run, no decision is re-made):

    zero-shot segmenter on C   p19_psma_hb_base.csv
    deployed agent f0 on C     p19_psma_hb_final.csv
    certified agent f1 on C    p19_psma_p19_final.csv
    naive / pseudo / uncal     p19_psma_{naive,pseudo,uncal}_final.csv
    fine-tuned segmenter (C)   p20_ft_p20_base.csv
    f1 / FT on B               p20B_psma_p19_final.csv / p20B_ft_hb_base.csv

The script asserts that the per-patient tables reproduce the locked aggregates
in phase20_eval.json before it computes anything. Everything here is post
hoc and descriptive (subgroups, distributions); no endpoint is added.

Writes paper/fig/fig_patients, fig_power, fig_dial_all, fig_l2_B (pdf+png)
and paper/posthoc_stats.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

R = Path(r"D:\data\reports")
REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "fig"
TSV = Path(r"D:\data\psma_nifti\PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv")
EPS = 0.01
rng = np.random.default_rng(0)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})
C_CERT, C_NAIVE, C_REF, C_GT, C_ALT = "#1f6fb2", "#c4452f", "#7a7a7a", "#2e8b57", "#e08a1e"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def table(name):
    rows = csv.DictReader(open(R / f"{name}.csv", newline=""))
    return {r["case"]: {k: float(v) for k, v in r.items() if k != "case"} for r in rows}


def harmed(new, ref):
    return (new["n_hit"] < ref["n_hit"]) or (ref["n_gt"] > 0 and new["dice"] < ref["dice"] - EPS)


def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ data --
split = json.loads((REPO / "reports" / "phase19_split.json").read_text())
meta = {}
for r in csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"):
    meta[(r["case_identifier_number"], r["study_date"].replace("-", ""))] = r


def info(case):
    pid, date, _ = case.rsplit("_", 2)
    return meta[(pid, date)]


T = {k: table(f) for k, f in {
    "M0": "p19_psma_hb_base", "HB": "p19_psma_hb_final", "P19": "p19_psma_p19_final",
    "NAIVE": "p19_psma_naive_final", "PSEUDO": "p19_psma_pseudo_final",
    "UNCAL": "p19_psma_uncal_final", "FT": "p20_ft_p20_base"}.items()}
TB = {"P19": table("p20B_psma_p19_final"), "FT": table("p20B_ft_hb_base")}
C = split["C"]
assert sorted(T["FT"]) == sorted(C) and sorted(TB["FT"]) == sorted(split["B"])

# reproduce the locked aggregates before using the tables
ev = json.loads((R / "phase20_eval.json").read_text())["arms"]
for arm in ("M0", "P19", "FT"):
    pos = [T[arm][c]["dice"] for c in C if T[arm][c]["n_gt"] > 0]
    assert abs(np.mean(pos) - ev[arm]["dice_pos"]) < 1e-9, arm
    assert int(sum(T[arm][c]["n_hit"] for c in C)) == ev[arm]["hits"], arm
assert sum(harmed(T["FT"][c], T["P19"][c]) for c in C) == ev["FT"]["harmed_vs_P19"]

vendors = sorted({info(c)["Manufacturer"] for c in split["A"] + split["B"] + C},
                 key=lambda v: sum(info(c)["Manufacturer"] == v for c in split["A"] + split["B"] + C))
VNAME = {v: f"vendor {i + 1}" for i, v in enumerate(vendors)}  # anonymised, by cohort size

# ------------------------------------------------------- subgroup table --
posC = [c for c in C if T["FT"][c]["n_gt"] > 0]
burden = np.array([T["FT"][c]["n_gt"] for c in posC])
q1, q2 = np.percentile(burden, [100 / 3, 200 / 3])


def group_of(c):
    g = {"all": True,
         "18F": info(c)["pet_radionuclide"] == "18F", "68Ga": info(c)["pet_radionuclide"] == "68Ga",
         "vendor 1": VNAME[info(c)["Manufacturer"]] == "vendor 1",
         "vendor 2": VNAME[info(c)["Manufacturer"]] == "vendor 2"}
    n = T["FT"][c]["n_gt"]
    g["burden low"] = 0 < n <= q1
    g["burden mid"] = q1 < n <= q2
    g["burden high"] = n > q2
    return g


sub = {}
for name in ("all", "18F", "68Ga", "vendor 1", "vendor 2", "burden low", "burden mid", "burden high"):
    cs = [c for c in C if group_of(c)[name]]
    pos = [c for c in cs if T["FT"][c]["n_gt"] > 0]
    d0 = [T["M0"][c]["dice"] for c in pos]
    d1 = [T["P19"][c]["dice"] for c in pos]
    d2 = [T["FT"][c]["dice"] for c in pos]
    gt = sum(T["FT"][c]["n_gt"] for c in cs)
    sub[name] = {
        "n": len(cs), "n_pos": len(pos), "lesions": int(gt),
        "dice_M0": float(np.mean(d0)), "dice_P19": float(np.mean(d1)), "dice_FT": float(np.mean(d2)),
        "dDice_FT_P19": boot_mean(np.array(d2) - np.array(d1)),
        "det_M0": sum(T["M0"][c]["n_hit"] for c in cs) / gt,
        "det_FT": sum(T["FT"][c]["n_hit"] for c in cs) / gt,
        "fp_M0": float(np.mean([T["M0"][c]["fp_ml"] for c in cs])),
        "fp_FT": float(np.mean([T["FT"][c]["fp_ml"] for c in cs])),
        "harmed_FT": int(sum(harmed(T["FT"][c], T["P19"][c]) for c in cs)),
    }

# ----------------------------------------------------- distributional --
dpos = np.array([T["FT"][c]["dice"] - T["P19"][c]["dice"] for c in posC])
dist = {
    "improve_gt_eps": int((dpos > EPS).sum()), "worse_gt_eps": int((dpos < -EPS).sum()),
    "ft_dice_median": float(np.median([T["FT"][c]["dice"] for c in posC])),
    "ft_dice_gt_05": int(sum(T["FT"][c]["dice"] > 0.5 for c in posC)),
    "p19_dice_gt_05": int(sum(T["P19"][c]["dice"] > 0.5 for c in posC)),
    "fp_gt_100_P19": int(sum(T["P19"][c]["fp_ml"] > 100 for c in C)),
    "fp_gt_100_FT": int(sum(T["FT"][c]["fp_ml"] > 100 for c in C)),
    "fp_zero_FT": int(sum(T["FT"][c]["fp_ml"] == 0 for c in C)),
    "lesion_det_M0": sum(T["M0"][c]["n_hit"] for c in C) / sum(T["M0"][c]["n_gt"] for c in C),
    "lesion_det_FT": sum(T["FT"][c]["n_hit"] for c in C) / sum(T["FT"][c]["n_gt"] for c in C),
    "lesions_gained": int(sum(max(0, T["FT"][c]["n_hit"] - T["P19"][c]["n_hit"]) for c in C)),
    "lesions_lost": int(sum(max(0, T["P19"][c]["n_hit"] - T["FT"][c]["n_hit"]) for c in C)),
    "patients_lost_any": int(sum(T["FT"][c]["n_hit"] < T["P19"][c]["n_hit"] for c in C)),
}

# Level-1 arms vs deployed agent: per-patient delta and harm
l1 = {}
for arm in ("P19", "NAIVE", "PSEUDO", "UNCAL"):
    dd = [T[arm][c]["dice"] - T["HB"][c]["dice"] for c in posC]
    l1[arm] = {"helped": int(sum(x > EPS for x in dd)), "hurt_dice": int(sum(x < -EPS for x in dd)),
               "changed": int(sum(abs(x) > 1e-9 for x in dd)),
               "harmed": int(sum(harmed(T[arm][c], T["HB"][c]) for c in C)),
               "lost_lesions": int(sum(max(0, T["HB"][c]["n_hit"] - T[arm][c]["n_hit"]) for c in C))}

# ---------------------------------------------------------------- power --
def p_val(k, n, d):
    return float(binom.cdf(k, n, d))


grid = np.linspace(0.005, 0.20, 400)
pl2 = np.array([p_val(4, 126, d) for d in grid])
d_star = float(grid[np.argmax(pl2 <= 0.025)])
need = {}
for d in (0.01, 0.02, 0.05):
    for k in (0, 4):
        n = next(n for n in range(10, 20000) if p_val(k, n, d) <= 0.025)
        need[f"delta={d},k={k}"] = n
power = {"level2_k": 4, "level2_n": 126, "delta_star_alpha_half": d_star, "n_needed_alpha_half": need}

# ------------------------------------------- harm-definition sensitivity --
# Re-count harm of the LOCKED arms under alternative harm definitions. No
# candidate is re-selected: at Level 1 a different definition would change which
# updates certify, which these tables cannot show (only deployed arms are stored).
def lost(new, ref):
    return ref["n_hit"] - new["n_hit"]


def dice_drop(new, ref, eps):
    return ref["n_gt"] > 0 and new["dice"] < ref["dice"] - eps


DEFS = {
    "primary":         lambda n, r: lost(n, r) >= 1 or dice_drop(n, r, 0.01),
    "lesion_only":     lambda n, r: lost(n, r) >= 1,
    "dice_only":       lambda n, r: dice_drop(n, r, 0.01),
    "eps0":            lambda n, r: lost(n, r) >= 1 or dice_drop(n, r, 0.0),
    "eps0.05":         lambda n, r: lost(n, r) >= 1 or dice_drop(n, r, 0.05),
    "lost2":           lambda n, r: lost(n, r) >= 2 or dice_drop(n, r, 0.01),
    "lost10pct":       lambda n, r: lost(n, r) > 0.10 * r["n_hit"] or dice_drop(n, r, 0.01),
}
assert DEFS["primary"](T["FT"][C[0]], T["P19"][C[0]]) == harmed(T["FT"][C[0]], T["P19"][C[0]])
sens = {}
for name, h in DEFS.items():
    row = {arm: int(sum(h(T[arm][c], T["HB"][c]) for c in C)) for arm in ("NAIVE", "PSEUDO", "UNCAL", "P19")}
    kB = int(sum(h(TB["FT"][c], TB["P19"][c]) for c in TB["FT"]))
    row.update({"FT_B": kB, "FT_B_p05": p_val(kB, len(TB["FT"]), 0.05),
                "FT_B_p10": p_val(kB, len(TB["FT"]), 0.10),
                "FT_C": int(sum(h(T["FT"][c], T["P19"][c]) for c in C))})
    sens[name] = row
assert sens["primary"]["NAIVE"] == l1["NAIVE"]["harmed"] and sens["primary"]["FT_B"] == 4
assert sens["primary"]["FT_C"] == ev["FT"]["harmed_vs_P19"]

json.dump({"subgroups_C": sub, "distribution_C": dist, "level1_vs_f0_C": l1, "power": power,
           "harm_definition_sensitivity": sens,
           "vendor_map_note": "vendor 1/2 ordered by cohort size; names withheld",
           "burden_tertiles": [float(q1), float(q2)]},
          open(Path(__file__).resolve().parent / "posthoc_stats.json", "w"), indent=1)


# ============================================================== figures ==
def fig_patients():
    """(a) Level 1: per-patient Dice change vs f0, naive vs certified (waterfall).
    (b) Level 2: per-patient Dice, f1 vs FT. (c) per-patient FP volume, f1 vs FT."""
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 1.85), gridspec_kw={"wspace": 0.38,
                                                                   "width_ratios": [1.45, 1, 1]})
    a = ax[0]
    for arm, col, lab, off in (("NAIVE", C_NAIVE, "naive", 0.0), ("P19", C_CERT, "certified (ours)", 0.0)):
        dd = np.array([T[arm][c]["dice"] - T["HB"][c]["dice"] for c in C])
        hh = np.array([harmed(T[arm][c], T["HB"][c]) for c in C])
        o = np.argsort(dd)
        x = np.arange(len(C))
        a.bar(x, dd[o], width=1.0, color=col, alpha=0.55 if arm == "NAIVE" else 0.95, lw=0,
              label=f"{lab}: {hh.sum()} harmed")
        a.scatter(x[hh[o]], dd[o][hh[o]], marker="x", s=11, color="k", lw=0.8, zorder=3,
                  label="harmed" if arm == "P19" else None)
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("test patients, sorted by $\\Delta$Dice")
    a.set_ylabel("$\\Delta$Dice vs. deployed $f_0$")
    a.set_xticks([])
    a.legend(frameon=False, fontsize=6.3, loc="upper left")
    a.set_title("(a) Level 1: per-patient change", fontsize=8)

    b = ax[1]
    hh = [c for c in C if harmed(T["FT"][c], T["P19"][c])]
    for nuc, mk in (("18F", "o"), ("68Ga", "s")):
        cs = [c for c in posC if info(c)["pet_radionuclide"] == nuc and c not in hh]
        b.scatter([T["P19"][c]["dice"] for c in cs], [T["FT"][c]["dice"] for c in cs], s=7, marker=mk,
                  facecolor="none" if nuc == "68Ga" else C_CERT, edgecolor=C_CERT, lw=0.6,
                  label=f"$^{{{nuc[:-1] if nuc == '18F' else nuc[:-2]}}}${nuc[-1] if nuc == '18F' else nuc[-2:]}")
    b.scatter([T["P19"][c]["dice"] for c in hh], [T["FT"][c]["dice"] for c in hh], marker="x", s=22,
              color=C_NAIVE, lw=1.0, label="harmed", zorder=3)
    b.plot([0, 1], [0, 1], color=C_REF, lw=0.7, ls=":")
    b.set_xlim(-0.02, 1); b.set_ylim(-0.02, 1)
    b.set_xlabel("Dice, Level-1 system $f_1$")
    b.set_ylabel("Dice, fine-tuned (ours)")
    b.legend(frameon=False, fontsize=6.0, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    b.set_title("(b) Level 2: Dice", fontsize=8)

    c_ = ax[2]
    f1 = np.array([T["P19"][c]["fp_ml"] for c in C]) + 1
    f2 = np.array([T["FT"][c]["fp_ml"] for c in C]) + 1
    c_.scatter(f1, f2, s=6, color=C_CERT, alpha=0.8, lw=0)
    c_.plot([1, 1e4], [1, 1e4], color=C_REF, lw=0.7, ls=":")
    c_.set_xscale("log"); c_.set_yscale("log")
    c_.set_xlim(0.8, 1e4); c_.set_ylim(0.8, 1e4)
    c_.set_xlabel("FP mL + 1, $f_1$")
    c_.set_ylabel("FP mL + 1, fine-tuned")
    c_.set_title("(c) Level 2: false positives", fontsize=8)
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_patients")


def fig_power():
    """(a) smallest certifiable delta vs n (k=0). (b) Level-2 p-value vs delta on B."""
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw={"wspace": 0.55})
    a = ax[0]
    n = np.arange(20, 1001)
    for m, ls, lab in ((1, "-", "$m=1$"), (2, "--", "$m=2$ (L2)"), (8, ":", "$m=8$ (L1)")):
        a.plot(n, 1 - (0.05 / m) ** (1 / n), ls, color=C_CERT, lw=1.0, label=lab)
    a.axvline(126, color=C_REF, lw=0.6)
    a.text(140, 0.058, "$n{=}126$", fontsize=6.3, color=C_REF, ha="left")
    a.set_xscale("log"); a.set_ylim(0, 0.13)
    a.set_xlabel("certification patients $n$")
    a.set_ylabel("smallest certifiable $\\delta$")
    a.legend(frameon=False, fontsize=5.8, loc="upper right")
    a.set_title("(a) cost of a guarantee", fontsize=8)

    b = ax[1]
    b.plot(grid, pl2, color=C_CERT, lw=1.1)
    b.axhline(0.025, color="k", lw=0.6, ls="--")
    b.text(0.198, 0.04, "$\\alpha/2$", fontsize=6.3, ha="right")
    b.axvspan(d_star, 0.2, color=C_CERT, alpha=0.08, lw=0)
    for d, ok in ((0.05, False), (0.10, True)):
        p = p_val(4, 126, d)
        b.scatter([d], [p], s=16, marker="o" if ok else "X", color=C_CERT if ok else C_NAIVE, zorder=3)
    b.text(0.058, 0.30, "$\\delta{=}0.05$: refused", fontsize=6.0, color=C_NAIVE)
    b.text(0.108, 0.004, "$\\delta{=}0.10$: certified", fontsize=6.0, color=C_CERT)
    b.set_yscale("log"); b.set_ylim(1e-4, 1.2); b.set_xlim(0.0, 0.2)
    b.set_xticks([0.05, 0.10, 0.15, 0.20]); b.set_xticklabels([".05", ".10", ".15", ".20"])
    b.set_xlabel("tolerance $\\delta$")
    b.set_ylabel("$p$-value ($k{=}4$ of 126)")
    b.set_title("(b) Level-2 decision", fontsize=8)
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_power")


def fig_dial_all():
    """Supplement: Level-1 dial on C for all three backbones."""
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 1.9), gridspec_kw={"wspace": 0.3})
    for tag, name, col, mk in (("psma", "nnU-Net ens.", C_CERT, "o"), ("swin_psma", "Swin", C_ALT, "s"),
                               ("nnuf0_psma", "nnU-Net single", C_GT, "^")):
        evj = json.loads((R / f"phase19_eval_{tag}.json").read_text())
        cur = [r for r in evj["curve"] if r["delta"] != "naive"]
        nv = next(r for r in evj["curve"] if r["delta"] == "naive")
        d = np.array([r["delta"] for r in cur])
        ax[0].plot(d, [r["C_sum"] for r in cur], "-" + mk, color=col, ms=3, lw=1, label=name)
        ax[0].scatter([0.215], [nv["C_sum"]], marker=mk, color=col, s=12)
        ax[1].errorbar(d, [r["C_k"] / 125 for r in cur],
                       yerr=[np.zeros(len(cur)), [r["C_ub95"] - r["C_k"] / 125 for r in cur]],
                       fmt=mk, color=col, ms=3, lw=0.8, capsize=1.5, label=name)
    ax[0].set_xlabel("tolerance $\\delta$  (right: naive)")
    ax[0].set_ylabel("benefit on $\\mathcal{C}$ ($\\sum\\Delta$Dice)")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].plot([0, 0.21], [0, 0.21], color=C_REF, lw=0.7, ls=":", label="$y=\\delta$")
    ax[1].set_xlabel("tolerance $\\delta$")
    ax[1].set_ylabel("harm rate on $\\mathcal{C}$ (95% UB)")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_dial_all")


def fig_l2_B():
    """Main paper (single column): Level 2 on the certification set B, per patient."""
    B = split["B"]
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.65), gridspec_kw={"wspace": 0.5})
    hh = [c for c in B if harmed(TB["FT"][c], TB["P19"][c])]
    pos = [c for c in B if TB["FT"][c]["n_gt"] > 0 and c not in hh]
    ax[0].scatter([TB["P19"][c]["dice"] for c in pos], [TB["FT"][c]["dice"] for c in pos], s=7,
                  color=C_CERT, lw=0, label="not harmed")
    ax[0].scatter([TB["P19"][c]["dice"] for c in hh], [TB["FT"][c]["dice"] for c in hh], marker="x",
                  s=24, color=C_NAIVE, lw=1.0, label=f"harmed ({len(hh)})")
    ax[0].plot([0, 1], [0, 1], color=C_REF, lw=0.7, ls=":")
    ax[0].set_xlabel("Dice, Level 1")
    ax[0].set_ylabel("Dice, fine-tuned")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(-0.05, 0.98), ncol=2, handletextpad=0.2, columnspacing=0.8)
    x = np.array([TB["P19"][c]["n_hit"] for c in B])
    y = np.array([TB["FT"][c]["n_hit"] for c in B])
    ax[1].scatter(x + 1, y + 1, s=7, color=C_CERT, lw=0)
    ax[1].scatter([TB["P19"][c]["n_hit"] + 1 for c in hh], [TB["FT"][c]["n_hit"] + 1 for c in hh],
                  marker="x", s=24, color=C_NAIVE, lw=1.0)
    ax[1].plot([1, 300], [1, 300], color=C_REF, lw=0.7, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("detected + 1, Level 1")
    ax[1].set_ylabel("detected + 1, fine-tuned")
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_l2_B")


if __name__ == "__main__":
    fig_patients()
    fig_power()
    fig_dial_all()
    fig_l2_B()
    print(json.dumps({"subgroups_C": sub, "distribution_C": dist, "level1_vs_f0_C": l1,
                      "power": power, "tertiles": [q1, q2]}, indent=1))
