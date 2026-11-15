#!/usr/bin/env python3
"""Phase XXIII evaluation, after reports/phase23_lock.md: H1, H2 and the
secondary endpoints on the follow-up studies.

See `reports/phase23_followup_confirmation_plan.md`. Inputs:
  p23_psmafu_hb_{base,final}.csv   M0 (base) and HB (final), from phase5_agent
  p23_psmafu_p19_final.csv         P19
  D:\\data\\cv\\ft_fu                FT maps, scored here at p > 0.5
Run with PETCT_RAW505 / PETCT_TOTALSEG set (labels live in the new copy).

Writes phase23_ft_fu.csv (FT per study) and phase23_eval.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import label_path
from phase5_eval import read
from phase21_safeswitch import score, harmed

R = Path(r"D:\data\reports")
REPO = Path(__file__).resolve().parent.parent
FT = Path(r"D:\data\cv\ft_fu")
ALPHA, NBOOT, SEED = 0.05, 2000, 2029


def ub(k, n):
    return 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))


def k_max(n, delta, a):
    k = -1
    while k + 1 <= n and binom.cdf(k + 1, n, delta) < a:
        k += 1
    return k


def ft_rows(cases):
    f = R / "phase23_ft_fu.csv"
    if f.exists():
        return read(R / "phase23_ft", "fu")
    out = {}
    for i, c in enumerate(cases, 1):
        z = np.load(FT / f"{c}.npz")
        p = z["p_bar"].astype(np.float32)
        vml = float(np.prod(z["spacing"])) / 1000.0
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(c)))) > 0
        assert p.shape == gt.shape, c
        g_lab, g_n = ndi.label(gt)
        out[c] = score(p > 0.5, gt, g_lab, g_n, vml)
        if i % 25 == 0 or i == len(cases):
            print(f"  FT scored {i}/{len(cases)}", flush=True)
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "dice", "n_hit", "n_gt", "fp_ml", "pred_ml"])
        w.writeheader()
        for c in sorted(out):
            w.writerow({"case": c, **out[c]})
    return read(R / "phase23_ft", "fu")


def patient(c):
    return c.rsplit("_", 2)[0]


def confirm(name, x, ref, cases, delta, alpha):
    pats = sorted({patient(c) for c in cases})
    hs = [c for c in cases if harmed(x[c], ref[c])]
    hp = sorted({patient(c) for c in hs})
    n, k = len(pats), len(hp)
    p = float(binom.cdf(k, n, delta))
    r = {"n_patients": n, "n_studies": len(cases), "k_patients": k, "k_studies": len(hs),
         "delta": delta, "alpha": alpha, "p": p, "met": p < alpha, "ub95": ub(k, n),
         "k_max_to_pass": k_max(n, delta, alpha), "harmed_studies": hs,
         "lost_hit_studies": [c for c in hs if x[c]["n_hit"] < ref[c]["n_hit"]]}
    print(f"  {name}: harmed {k}/{n} patients ({len(hs)}/{len(cases)} studies), "
          f"p(R>{delta}) = {p:.4f}, UB {r['ub95']:.3f}, passes iff k <= {r['k_max_to_pass']} "
          f"-> {'MET' if r['met'] else 'NOT MET'}")
    return r


def cluster_boot(a, b, cases, rng):
    """Dice a - b on tumour-positive studies; per patient the mean, patients resampled."""
    by = {}
    for c in cases:
        if b[c]["n_gt"] > 0:
            by.setdefault(patient(c), []).append(a[c]["dice"] - b[c]["dice"])
    d = np.array([np.mean(v) for v in by.values()])
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    return {"n_patients": len(d), "delta": float(d.mean()),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p_one_sided": float(max((bs <= 0).mean(), 1 / NBOOT))}


def describe(x, cases):
    pos = [c for c in cases if x[c]["n_gt"] > 0]
    neg = [c for c in cases if x[c]["n_gt"] == 0]
    return {"dice_pos_study": float(np.mean([x[c]["dice"] for c in pos])),
            "fp_ml": float(np.mean([x[c]["fp_ml"] for c in cases])),
            "hits": int(sum(x[c]["n_hit"] for c in cases)),
            "gt_lesions": int(sum(x[c]["n_gt"] for c in cases)),
            "n_pos": len(pos), "n_neg": len(neg),
            "neg_cleared": int(sum(x[c]["fp_ml"] == 0 for c in neg))}


def main() -> None:
    assert (REPO / "reports" / "phase23_lock.md").exists(), "commit the lock first"
    cases = sorted(Path(REPO / "reports" / "cases_psma_fu.txt").read_text().split())
    excl = json.loads((R / "phase23_exclusions.json").read_text())
    cases = [c for c in cases if c not in excl]
    arms = {"M0": read(R / "p23_psmafu_hb", "base"), "HB": read(R / "p23_psmafu_hb", "final"),
            "P19": read(R / "p23_psmafu_p19", "final")}
    arms["FT"] = ft_rows(cases)
    for k, v in arms.items():
        assert set(cases) <= set(v), k
    for c in cases:  # the M0 of both agent runs is the same mask
        assert arms["M0"][c]["n_hit"] == read(R / "p23_psmafu_p19", "base")[c]["n_hit"]
    sp = json.loads((REPO / "reports" / "phase19_split.json").read_text())
    grp = {patient(c): s for s in "BC" for c in sp[s]}
    rep = {"n_studies": len(cases), "n_patients": len({patient(c) for c in cases}),
           "excluded": excl}
    print(f"[F] {rep['n_studies']} studies of {rep['n_patients']} patients")

    print("primary confirmations")
    h1 = confirm("H1 Level 2 (FT vs P19, delta 0.10)", arms["FT"], arms["P19"], cases, 0.10, ALPHA)
    h2 = confirm("H2 Level 1 (P19 vs HB, delta 0.05)", arms["P19"], arms["HB"], cases, 0.05, ALPHA)
    rep["H1"], rep["H2"] = h1, h2
    ps = sorted([("H1", h1["p"]), ("H2", h2["p"])], key=lambda t: t[1])
    holm, stop = {}, False
    for i, (nm, p) in enumerate(ps):
        holm[nm] = (not stop) and p < ALPHA / (2 - i)
        stop = stop or not holm[nm]
    rep["holm"] = holm
    print(f"  Holm over H1, H2: {holm}")

    print("secondary")
    rep["FT_vs_P19_delta05"] = confirm("FT vs P19 at delta 0.05", arms["FT"], arms["P19"],
                                       cases, 0.05, ALPHA)
    rng = np.random.default_rng(SEED)
    rep["benefit"] = {}
    for a, b in (("FT", "P19"), ("FT", "M0"), ("P19", "HB"), ("P19", "M0")):
        r = cluster_boot(arms[a], arms[b], cases, rng)
        rep["benefit"][f"{a}-{b}"] = r
        print(f"  Dice {a}-{b}: {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"p1 = {r['p_one_sided']:.4f} ({r['n_patients']} patients)")
    rep["arms"] = {k: describe(v, cases) for k, v in arms.items()}
    for k, v in rep["arms"].items():
        print(f"  {k:4s} Dice/study {v['dice_pos_study']:.4f} FP {v['fp_ml']:7.1f} mL "
              f"hits {v['hits']}/{v['gt_lesions']} neg cleared {v['neg_cleared']}/{v['n_neg']}")
    rep["subsets"] = {}
    for s in "BC":
        cs = [c for c in cases if grp[patient(c)] == s]
        rep["subsets"][s] = {
            "H1": confirm(f"  {s}-patients H1", arms["FT"], arms["P19"], cs, 0.10, ALPHA),
            "H2": confirm(f"  {s}-patients H2", arms["P19"], arms["HB"], cs, 0.05, ALPHA),
            "FT-P19": cluster_boot(arms["FT"], arms["P19"], cs, rng)}
    (R / "phase23_eval.json").write_text(json.dumps(rep, indent=1, default=float))
    print("written phase23_eval.json")


if __name__ == "__main__":
    main()
