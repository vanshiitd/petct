#!/usr/bin/env python3
"""The work plan's section-23 numbers for Experiment 4, plus the Q score.

Per variant, against M0:
  * share of cases improved / unchanged / degraded in Dice, mean delta Dice,
    mean delta HD95
  * tumour-free patients cleaned (had a false positive under M0, none under M1)
  * ground-truth lesions lost and recovered, by size bin -- computed from the
    masks on disk rather than from the decision bookkeeping, so it is an
    independent check on the whole chain
  * paired bootstrap of delta Dice over tumour-positive patients

"Unchanged" means a Dice change under 1e-6, not merely small: a case the
refinement never touched should land there exactly, and any case that moves at
all is counted as improved or degraded however little it moved.

The case quality score Q (eq. 16) is the volume-weighted mean P(not FP) over
the components M1 keeps. It is reported against Dice and false-positive volume
next to Phase II's uncertainty-only case score for comparison.

    python scripts/exp4_numbers.py --variants Acons Bcons ABcons ABRcons ABdice
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi, stats
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import GROUPS, load, matrix     # noqa: E402

REP = Path(r"D:\data\reports")
PRED = Path(r"D:\data\nnunet\predictions")
GT = Path(r"D:\data\nnunet\gt_v2_test")
BINS = [(0, 0.1, "<0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.0, "0.5-1"), (1.0, 1e9, ">=1")]


def read_eval(tag: str) -> dict:
    path = REP / f"eval_{tag}.csv" if tag != "M0" else REP / "evalM0_test.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def fnum(v):
    try:
        return float(v)
    except ValueError:
        return float("nan")


def lesion_flow(variant_dir: Path, m0_dir: Path) -> dict:
    """GT lesions hit by M0 and by M1, per size bin, straight from the masks."""
    counts = {n: {"m0_hit": 0, "m1_hit": 0, "lost": 0, "recovered": 0, "total": 0,
                  "lost_ml": 0.0, "rec_ml": 0.0}
              for *_, n in BINS}
    for gt_file in sorted(GT.glob("*.nii.gz")):
        case = gt_file.name[:-7]
        g_img = sitk.ReadImage(str(gt_file))
        gt = sitk.GetArrayFromImage(g_img) > 0
        if not gt.any():
            continue
        sx, sy, sz = g_img.GetSpacing()
        vml = sx * sy * sz / 1000.0
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(m0_dir / gt_file.name))) > 0
        b = sitk.GetArrayFromImage(sitk.ReadImage(str(variant_dir / gt_file.name))) > 0
        lab, n = ndi.label(gt)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        ha = np.bincount(lab[a].ravel(), minlength=n + 1) if a.any() else np.zeros(n + 1, int)
        hb = np.bincount(lab[b].ravel(), minlength=n + 1) if b.any() else np.zeros(n + 1, int)
        for cid in range(1, n + 1):
            v = sizes[cid] * vml
            name = next(nm for lo, hi, nm in BINS if lo <= v < hi)
            c = counts[name]
            c["total"] += 1
            c["m0_hit"] += int(ha[cid] > 0)
            c["m1_hit"] += int(hb[cid] > 0)
            if ha[cid] > 0 and hb[cid] == 0:
                c["lost"] += 1
                c["lost_ml"] += v
            if ha[cid] == 0 and hb[cid] > 0:
                c["recovered"] += 1
                c["rec_ml"] += v
    return counts


def q_scores() -> dict:
    """Volume-weighted mean P(not FP) over the components M0 predicts."""
    models = {}
    for kind in ("pred",):
        rows_v, yv, _, _ = load(REP / "cand_val.csv", kind)
        feats = GROUPS["d all"]
        Xv, med, _ = matrix(rows_v, feats, None)
        mu, sd = Xv.mean(0), Xv.std(0); sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=5000, C=1.0).fit((Xv - mu) / sd, yv)
        models[kind] = (clf, mu, sd, med, feats)
    clf, mu, sd, med, feats = models["pred"]
    rows, _, _, _ = load(REP / "cand_test.csv", "pred")
    X, _, _ = matrix(rows, feats, med)
    p_fp = clf.predict_proba((X - mu) / sd)[:, 1]
    per_case: dict[str, list] = {}
    for r, p in zip(rows, p_fp):
        per_case.setdefault(r["case"], []).append((float(r["volume_ml"]), 1.0 - p))
    return {c: (sum(v * q for v, q in L) / sum(v for v, _ in L)) if sum(v for v, _ in L)
            else 1.0 for c, L in per_case.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()

    m0 = read_eval("M0")
    print(f"{'variant':12s} {'Dice':>7s} {'dDice':>8s} {'impr':>6s} {'unch':>6s} "
          f"{'degr':>6s} {'dHD95':>8s} {'cleaned':>8s} {'lost':>5s} {'rec':>4s}")
    print("-" * 82)
    out = {}
    for tag in args.variants:
        ev = read_eval(tag)
        cases = sorted(set(m0) & set(ev))
        pos = [c for c in cases if m0[c]["gt_positive"] == "1"]
        d0 = np.array([fnum(m0[c]["dice"]) for c in pos])
        d1 = np.array([fnum(ev[c]["dice"]) for c in pos])
        dd = d1 - d0
        impr = int((dd > 1e-6).sum()); degr = int((dd < -1e-6).sum())
        unch = len(dd) - impr - degr
        h0 = np.array([fnum(m0[c]["hd95_mm"]) for c in pos])
        h1 = np.array([fnum(ev[c]["hd95_mm"]) for c in pos])
        ok = np.isfinite(h0) & np.isfinite(h1)
        cleaned = sum(1 for c in cases if m0[c]["gt_positive"] == "0"
                      and fnum(m0[c]["fp_volume_ml"]) > 0
                      and fnum(ev[c]["fp_volume_ml"]) == 0)
        flow = lesion_flow(PRED / f"test_{tag}_mapped", PRED / "test_M0_mapped")
        lost = sum(v["lost"] for v in flow.values())
        rec = sum(v["recovered"] for v in flow.values())
        rng = np.random.default_rng(2026)
        boot = [float(dd[rng.integers(0, len(dd), len(dd))].mean()) for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{tag:12s} {d1.mean():7.4f} {dd.mean():+8.4f} {impr:>6d} {unch:>6d} "
              f"{degr:>6d} {np.mean(h1[ok]-h0[ok]):+8.2f} {cleaned:>8d} {lost:>5d} {rec:>4d}")
        out[tag] = {"dice": float(d1.mean()), "d_dice": float(dd.mean()),
                    "ci": [float(lo), float(hi)], "improved": impr, "unchanged": unch,
                    "degraded": degr, "d_hd95": float(np.mean(h1[ok] - h0[ok])),
                    "cleaned": cleaned, "lesion_flow": flow,
                    "n_pos": len(pos)}
        print(f"{'':12s} paired bootstrap dDice 95% CI [{lo:+.4f}, {hi:+.4f}] "
              f"{'excludes 0' if (lo > 0) == (hi > 0) else 'straddles 0'}")

    print(f"\nground-truth lesions lost / recovered by size, per variant")
    print(f"{'variant':12s} " + " ".join(f"{n:>16s}" for *_, n in BINS))
    for tag in args.variants:
        f = out[tag]["lesion_flow"]
        print(f"{tag:12s} " + " ".join(
            f"{('-' + str(f[n]['lost']) + ' / +' + str(f[n]['recovered'])):>16s}"
            for *_, n in BINS))

    # --- case-level Q -------------------------------------------------
    q = q_scores()
    ev0 = read_eval("M0")
    cs = [c for c in ev0 if c in q]
    pos = [c for c in cs if ev0[c]["gt_positive"] == "1"]
    qd = np.array([q[c] for c in pos]); dd = np.array([fnum(ev0[c]["dice"]) for c in pos])
    qa = np.array([q[c] for c in cs]); fp = np.array([fnum(ev0[c]["fp_volume_ml"]) for c in cs])
    print(f"\ncase quality score Q (volume-weighted mean P(not FP)), test")
    print(f"  Q vs Dice      Spearman {stats.spearmanr(qd, dd).statistic:+.4f} "
          f"(n={len(pos)} tumour-positive)")
    print(f"  Q vs FP volume Spearman {stats.spearmanr(qa, fp).statistic:+.4f} "
          f"(n={len(cs)} all cases)")
    print(f"  for comparison, Phase II uncertainty-only case score: "
          f"rho 0.74 vs 1-Dice, 0.72 vs FP volume")
    out["Q"] = {"spearman_dice": float(stats.spearmanr(qd, dd).statistic),
                "spearman_fp": float(stats.spearmanr(qa, fp).statistic)}
    (REP / "exp4_numbers.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {REP / 'exp4_numbers.json'}")


if __name__ == "__main__":
    main()
