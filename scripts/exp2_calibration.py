#!/usr/bin/env python3
"""Voxel-level AUROC and calibration of P_bar, for val and test.

AUROC comes from the stratified voxel samples; calibration comes from the
histograms, where the bins are coarse and the saturation that makes histogram
AUROC unusable does not bite.

A note on what "calibration" means here. P_bar is a per-voxel foreground
probability, so the natural check is whether voxels with P_bar near 0.3 are
foreground about 30% of the time. Over the whole body that question is almost
vacuous -- the base rate is 0.1% and nearly every voxel sits in the bottom bin
-- so calibration is reported in the near-tumour region as well, where the base
rate is 12% and the diagram has something to show.

ECE is computed with the bin weights of the region in question. Temperature
scaling fits one scalar T on val by minimising val negative log-likelihood over
the binned counts, which is exactly equivalent to fitting it on the voxels since
the likelihood only depends on the counts per bin, and then applies that same T
to test. A T above 1 means the model is overconfident.

    python scripts/exp2_calibration.py --val D:\\data\\reports\\exp2_val \\
        --test D:\\data\\reports\\exp2_test --out D:\\data\\reports\\exp2_calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

U_LABELS = {"u_ent": "U_ent (= confidence baseline)", "u_epi": "U_epi", "u_stab": "U_stab"}
NCAL = 20


def voxel_auroc(samples_path: Path) -> dict:
    z = np.load(samples_path)
    out = {}
    for region in ("body", "near"):
        e = z[f"{region}__err"].astype(bool)
        ne, nn = int(e.sum()), int((~e).sum())
        denom = float(ne) * float(nn)
        # The sample is stratified -- every error kept, non-errors thinned -- so
        # the error rate IN the sample is not the error rate in the region. Undo
        # the thinning to report the population rate, otherwise the body region
        # looks 150x more error-prone than it is. AUROC itself is a two-sample
        # statistic and needs no such correction.
        q = json.loads(str(z["keep_nonerr"]))[region]
        out[region] = {"n_err": ne, "n_ok": nn, "keep_nonerr": q,
                       "err_rate_population": ne / (ne + nn / q)}
        for k in U_LABELS:
            v = z[f"{region}__{k}"].astype(np.float32)
            out[region][k] = float(stats.mannwhitneyu(v[e], v[~e]).statistic) / denom
        # equal-weight combined U, z-scored on this region's own sample
        zs = []
        for k in U_LABELS:
            v = z[f"{region}__{k}"].astype(np.float32)
            zs.append((v - v.mean()) / (v.std() or 1.0))
        c = np.sum(zs, axis=0)
        out[region]["combined_equal"] = (
            float(stats.mannwhitneyu(c[e], c[~e]).statistic) / denom)
    return out


def reliability(hist_path: Path, region: str, temperature: float = 1.0):
    """Observed foreground rate per P_bar bin, optionally after temperature T."""
    h = np.load(hist_path)
    c = h[f"{region}__p_bar"]          # [0] non-error, [1] error, over 2048 p bins
    nbins = c.shape[1]
    centres = (np.arange(nbins) + 0.5) / nbins
    # An error voxel is one where (P_bar > 0.5) disagrees with truth, so the
    # truth label per bin follows from the bin's own side of 0.5.
    above = centres > 0.5
    fg = np.where(above, c[0], c[1]).astype(np.float64)      # truth = foreground
    bg = np.where(above, c[1], c[0]).astype(np.float64)      # truth = background
    p = centres
    if temperature != 1.0:
        lg = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
        p = 1.0 / (1.0 + np.exp(-lg / temperature))
    edges = np.linspace(0.0, 1.0, NCAL + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, NCAL - 1)
    out = []
    tot = fg.sum() + bg.sum()
    ece = 0.0
    for b in range(NCAL):
        m = idx == b
        n = fg[m].sum() + bg[m].sum()
        if n == 0:
            out.append({"bin": b, "n": 0, "conf": float("nan"), "freq": float("nan")})
            continue
        conf = float((p[m] * (fg[m] + bg[m])).sum() / n)
        freq = float(fg[m].sum() / n)
        ece += (n / tot) * abs(conf - freq)
        out.append({"bin": b, "n": int(n), "conf": conf, "freq": freq})
    return out, float(ece), float(tot)


def fit_temperature(hist_path: Path, region: str) -> float:
    h = np.load(hist_path)
    c = h[f"{region}__p_bar"]
    nbins = c.shape[1]
    centres = (np.arange(nbins) + 0.5) / nbins
    above = centres > 0.5
    fg = np.where(above, c[0], c[1]).astype(np.float64)
    bg = np.where(above, c[1], c[0]).astype(np.float64)
    keep = (fg + bg) > 0
    lg = np.log(np.clip(centres, 1e-7, 1 - 1e-7) / np.clip(1 - centres, 1e-7, 1 - 1e-7))
    lg, fg, bg = lg[keep], fg[keep], bg[keep]
    best, bestT = None, 1.0
    for T in np.concatenate([np.arange(0.2, 3.01, 0.02), np.arange(3.1, 10.1, 0.1)]):
        q = 1.0 / (1.0 + np.exp(-lg / T))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        nll = -(fg * np.log(q) + bg * np.log(1 - q)).sum()
        if best is None or nll < best:
            best, bestT = nll, float(T)
    return bestT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True, help="prefix, e.g. ..._val")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res = {}
    for split, pref in (("val", args.val), ("test", args.test)):
        res[split] = {"auroc": voxel_auroc(
            pref.with_name(pref.name + "_samples.npz"))}

    print("voxel-level AUROC for predicting an error voxel")
    print(f"{'split':5s} {'region':6s} {'U_ent':>8s} {'U_epi':>8s} {'U_stab':>8s} "
          f"{'comb':>8s}   {'err rate':>9s}")
    for split in ("val", "test"):
        for r in ("body", "near"):
            a = res[split]["auroc"][r]
            rate = a["err_rate_population"]
            print(f"{split:5s} {r:6s} {a['u_ent']:8.4f} {a['u_epi']:8.4f} "
                  f"{a['u_stab']:8.4f} {a['combined_equal']:8.4f}   {rate:9.3%}")
    print("\nU_ent is also the confidence baseline: max(P, 1-P) is a monotone\n"
          "function of it, so the two have identical AUROC by construction.")

    print("\ncalibration of P_bar")
    for region in ("body", "near"):
        T = fit_temperature(args.val.with_name(args.val.name + "_hist.npz"), region)
        print(f"\n  region '{region}': temperature fitted on val T = {T:.2f}"
              f"  ({'overconfident' if T > 1 else 'underconfident'})")
        for split, pref in (("val", args.val), ("test", args.test)):
            h = pref.with_name(pref.name + "_hist.npz")
            raw, ece_raw, n = reliability(h, region, 1.0)
            cal, ece_cal, _ = reliability(h, region, T)
            print(f"    {split:5s} ECE {ece_raw:.4f} -> {ece_cal:.4f} after T "
                  f"({'better' if ece_cal < ece_raw else 'WORSE'}), {n:,.0f} voxels")
            res[split].setdefault("calibration", {})[region] = {
                "temperature": T, "ece_raw": ece_raw, "ece_cal": ece_cal,
                "bins_raw": raw, "bins_cal": cal}
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
