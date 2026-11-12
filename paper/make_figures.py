#!/usr/bin/env python3
"""Figures for the paper, from the locked result files only.

    python paper/make_figures.py

Reads D:\\data\\reports (phase19_eval_*.json, phase20_eval.json, the C maps) and
writes paper/fig/*.pdf (+ .png previews). No number here is retyped: every
plotted value is read from the JSON the evaluation scripts wrote.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.patches import FancyBboxPatch

R = Path(r"D:\data\reports")
OUT = Path(__file__).resolve().parent / "fig"
RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
MAPS = {"M0": Path(r"D:\data\cv\psma"), "FT": Path(r"C:\psma_maps\ft")}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})
C_CERT, C_NAIVE, C_REF, C_GT = "#1f6fb2", "#c4452f", "#7a7a7a", "#2e8b57"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ dial --
def fig_dial():
    ev = json.loads((R / "phase19_eval_psma.json").read_text())
    curve = [r for r in ev["curve"] if r["delta"] != "naive"]
    naive = next(r for r in ev["curve"] if r["delta"] == "naive")
    nC = 125
    d = np.array([r["delta"] for r in curve])
    ben = np.array([r["C_sum"] for r in curve])
    rate = np.array([r["C_k"] / nC for r in curve])
    ubs = np.array([r["C_ub95"] for r in curve])

    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw={"wspace": 0.55})
    a = ax[0]
    a.plot(d, ben, "-o", color=C_CERT, ms=3, lw=1.1, label="certified")
    a.axhline(naive["C_sum"], color=C_NAIVE, ls="--", lw=0.9, label="naive")
    a.set_xlabel(r"tolerance $\delta$")
    a.set_ylabel(r"benefit on $\mathcal{C}$ ($\sum\Delta$Dice)")
    a.set_xticks([0.05, 0.10, 0.15, 0.20])
    a.set_xticklabels([".05", ".10", ".15", ".20"], fontsize=6.5)
    a.legend(frameon=False, fontsize=6.5, loc="lower right")
    a.set_title("(a) benefit", fontsize=8)

    b = ax[1]
    b.plot([0, 0.21], [0, 0.21], color=C_REF, lw=0.7, ls=":", label=r"$y=\delta$")
    b.errorbar(d, rate, yerr=[np.zeros_like(rate), ubs - rate], fmt="o", color=C_CERT,
               ms=3, lw=0.9, capsize=1.5, label="realised (95% UB)")
    b.axhline(naive["C_k"] / nC, color=C_NAIVE, ls="--", lw=0.9)
    b.set_xlim(0.0, 0.21)
    b.set_ylim(0.0, 0.21)
    b.set_xlabel(r"tolerance $\delta$")
    b.set_ylabel(r"harm rate on $\mathcal{C}$")
    b.set_xticks([0.05, 0.10, 0.15, 0.20])
    b.set_xticklabels([".05", ".10", ".15", ".20"], fontsize=6.5)
    b.legend(frameon=False, fontsize=6, loc="upper left")
    b.set_title("(b) harm", fontsize=8)
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_dial")


# -------------------------------------------------------------- validity --
def fig_validity():
    rows = []
    for tag, name in (("psma", "nnU-Net ens."), ("swin_psma", "Swin"), ("nnuf0_psma", "nnU-Net single")):
        ev = json.loads((R / f"phase19_eval_{tag}.json").read_text())
        v = next(r for r in ev["validity"] if abs(r["delta"] - 0.05) < 1e-9)
        rows.append((name, v["ltt"]["violation_rate"], v["naive"]["violation_rate"]))
    fig, a = plt.subplots(figsize=(3.35, 1.45))
    x = np.arange(len(rows))
    w = 0.36
    lt = [r[1] for r in rows]
    nv = [r[2] for r in rows]
    a.bar(x - w / 2, lt, w, color=C_CERT, label="certified (LTT)")
    a.bar(x + w / 2, nv, w, color=C_NAIVE, label="naive selection")
    a.axhline(0.05, color="k", lw=0.7, ls="--")
    a.text(0.5, 0.075, r"$\alpha=0.05$", fontsize=6.5, ha="center")
    for i, (l, n) in enumerate(zip(lt, nv)):
        a.text(i - w / 2, 0.11, f"{l:.3f}", ha="center", fontsize=6, color=C_CERT)
        a.text(i + w / 2, n + 0.02, f"{n:.3f}", ha="center", fontsize=6)
    a.set_xticks(x)
    a.set_xticklabels([r[0] for r in rows])
    a.set_ylim(0, 1.12)
    a.set_ylabel(r"P(realised harm $>\delta$)")
    a.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    a.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_validity")


# ----------------------------------------------------------- qualitative --
CASES = [  # (case, why) -- picked by rank of FT-M0 dDice on C, plus one harmed case
    ("PSMA_d4b471bab61342ff_20030822_24262548", "median gain"),
    ("PSMA_b361fb9c2455deb9_19990825_07609103", "upper quartile"),
    ("PSMA_1844105bd00f422d_20021211_83008799", "large FP removed"),
    ("PSMA_8b9bd3dbbb1553f1_20011226_26804343", "counted as harmed"),
]


def mip(a, axis=1):
    return a.max(axis=axis)


def load(case):
    pet = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "imagesTs" / f"{case}_0000.nii.gz")))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "labelsTs_prep" / f"{case}.nii.gz"))) > 0
    m = {k: np.load(p / f"{case}.npz")["p_bar"].astype(np.float32) >= 0.5 for k, p in MAPS.items()}
    return pet, gt, m


def pad_to(a, shape, fill):
    out = np.full(shape, fill, dtype=a.dtype)
    z0, x0 = (shape[0] - a.shape[0]) // 2, (shape[1] - a.shape[1]) // 2
    out[z0:z0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def fig_qual():
    data = [load(c) for c, _ in CASES]
    H = max(mip(d[0]).shape[0] for d in data)
    W = max(mip(d[0]).shape[1] for d in data)
    fig, axs = plt.subplots(1, 12, figsize=(6.9, 2.3), gridspec_kw={"wspace": 0.03})
    titles = ["GT", "zero-shot", "ours"]
    for j, ((case, why), (pet, gt, m)) in enumerate(zip(CASES, data)):
        p = mip(pet)[::-1]  # coronal MIP, head up
        lo, hi = np.percentile(p, [1, 99.7])
        p = pad_to(p, (H, W), float(lo))
        for i, (lab, msk, col) in enumerate((("GT", gt, C_GT), ("M0", m["M0"], C_NAIVE),
                                             ("FT", m["FT"], C_CERT))):
            a = axs[3 * j + i]
            a.imshow(p, cmap="gray_r", vmin=lo, vmax=hi, aspect=3.0 / 2.0364,
                     interpolation="bilinear")
            mm = pad_to(mip(msk)[::-1].astype(float), (H, W), 0.0)
            a.imshow(np.ma.masked_where(mm == 0, mm), cmap=matplotlib.colors.ListedColormap([col]),
                     alpha=0.75, aspect=3.0 / 2.0364, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_title(titles[i], fontsize=6.5, pad=2)
        axs[3 * j + 1].text(0.5, -0.06, f"({'abcd'[j]}) {why}", transform=axs[3 * j + 1].transAxes,
                            ha="center", va="top", fontsize=7)
    save(fig, "fig_qual")


# -------------------------------------------------------------- overview --
def box(a, x, y, w, h, text, fc, ec="none", fs=7, bold=False):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                               fc=fc, ec=ec, lw=0.8))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
           weight="bold" if bold else "normal", linespacing=1.15)


def arrow(a, x0, y0, x1, y1, text="", col="k"):
    a.annotate("", xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col, mutation_scale=8))
    if text:
        a.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.035, text, ha="center", fontsize=6.2, color=col)


def fig_overview():
    e20 = json.loads((R / "phase20_eval.json").read_text())["arms"]
    fig = plt.figure(figsize=(6.9, 1.75))
    a = fig.add_axes([0, 0, 1, 1])
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    box(a, 0.005, 0.56, 0.15, 0.36, "Deployed system\n$f_0$ (source: FDG)\nsegmenter + agent", "#e8eef6")
    box(a, 0.005, 0.08, 0.15, 0.36, "New domain (PSMA)\n377 patients, split\n"
        r"$\mathcal{A}|\mathcal{B}|\mathcal{C}$ = 126|126|125", "#f3efe6")
    box(a, 0.215, 0.56, 0.17, 0.36, r"$\mathcal{A}$: adapt" "\n" r"propose updates" "\n"
        r"$\Lambda=\{\lambda_1,\dots,\lambda_m\}$", "#eef3ea")
    box(a, 0.44, 0.56, 0.2, 0.36, r"$\mathcal{B}$: certify (LTT)" "\n"
        r"$H_\lambda: R(\lambda)>\delta$" "\n" r"FWER $\leq\alpha$", "#fff4d6", ec="#d9a400")
    box(a, 0.70, 0.70, 0.13, 0.22, "deploy $\\hat\\lambda$", "#dfeefb", ec=C_CERT)
    box(a, 0.70, 0.40, 0.13, 0.22, "roll back\nto $f_0$", "#f6e1dd", ec=C_NAIVE)
    box(a, 0.865, 0.40, 0.13, 0.52, r"$\mathcal{C}$: locked" "\n" "test, read" "\n" "once", "#eeeeee")
    arrow(a, 0.155, 0.74, 0.215, 0.74)
    arrow(a, 0.155, 0.26, 0.24, 0.55, col="#666")
    arrow(a, 0.385, 0.74, 0.44, 0.74)
    arrow(a, 0.64, 0.80, 0.70, 0.81, "certified")
    arrow(a, 0.64, 0.66, 0.70, 0.52, "none", col=C_NAIVE)
    arrow(a, 0.83, 0.81, 0.865, 0.75)
    arrow(a, 0.83, 0.51, 0.865, 0.55)
    ft, m0 = e20["FT"], e20["M0"]
    a.text(0.215, 0.24,
           "risk  $R(\\lambda)=\\Pr[$patient harmed by $\\lambda$ vs. $f_0]$,\n"
           "harmed = a lesion lost, or Dice drop $>0.01$\n"
           f"locked $\\mathcal{{C}}$: Dice {m0['dice_pos']:.2f}$\\rightarrow${ft['dice_pos']:.2f}, "
           f"FP {m0['fp_ml']:.0f}$\\rightarrow${ft['fp_ml']:.1f} mL, harm {ft['harmed_vs_P19']}/125",
           fontsize=6.6, va="center", linespacing=1.4)
    save(fig, "fig_overview")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_dial()
    fig_validity()
    fig_overview()
    fig_qual()
    print("written", sorted(p.name for p in OUT.glob("*.pdf")))
