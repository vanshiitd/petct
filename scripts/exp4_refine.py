#!/usr/bin/env python3
"""Experiment 4: a single-pass refinement M0 -> M1.

Two actions, both driven by the Experiment 3 detector 'd all' (logistic
regression on model + metabolic + anatomy + CT + shape + position, fitted on
val):

  A (remove)  drop a predicted component when P(FP) > tau_A
  B (recover) add a sub-threshold candidate when P(recoverable) > tau_B

and a "reflection" variant that applies a case's changes only when the case
quality score Q improves, Q being the volume-weighted mean P(not FP) over the
components that survive -- the plan's eq. 16.

Thresholds are tuned on val at two operating points:

  dice         maximise mean Dice over tumour-positive val cases
  conservative minimise the number of tumour-free val patients carrying any
               false positive, subject to losing at most 1% of true-positive
               lesion volume

Tuning does not write or re-read a single mask. Each case is reduced once to
component bookkeeping -- per component its voxel count, its overlap with the
ground truth, and which ground-truth components it touches -- from which Dice,
false-positive volume and false-negative volume follow exactly for any choice
of thresholds. That makes a 2-D threshold sweep free, and it is exact rather
than approximate because every action here keeps or drops whole components.

Component identity matches `build_candidates.py` exactly: same probability map,
same thresholds, same 6-connectivity, same minimum size, so `ndi.label` returns
the same ids and a decision made on a CSV row applies to the right voxels.
Predicted components too small to be candidates are never removed; they remain
part of M0 as they were.

    python scripts/exp4_refine.py --val-cand reports/cand_val.csv \\
        --test-cand reports/cand_test.csv --out-dir D:\\data\\nnunet\\predictions
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import GROUPS, load, matrix     # noqa: E402

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
UNC = Path(r"D:\data\uncertainty_m0")
T_LOW, MIN_VOX = 0.1, 5


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(case)


def bookkeep(case: str, split: str) -> dict:
    """Per-component sizes, ground-truth overlap and touched GT ids."""
    z = np.load(UNC / split / f"{case}.npz")
    P = z["p_bar"].astype(np.float32)
    spacing = tuple(float(v) for v in z["spacing"])
    vml = float(np.prod(spacing)) / 1000.0
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
    m0 = P > 0.5
    g_lab, g_n = ndi.label(gt)
    g_sizes = np.bincount(g_lab.ravel(), minlength=g_n + 1)

    def describe(lab, n, sizes, ids):
        out = {}
        for cid in ids:
            m = lab == cid
            inter = int(np.logical_and(m, gt).sum())
            touched = (set(np.unique(g_lab[m]).tolist()) - {0}) if inter else set()
            out[cid] = {"size": int(sizes[cid]), "inter": inter, "gt": touched}
        return out

    p_lab, p_n = ndi.label(m0)
    p_sizes = np.bincount(p_lab.ravel(), minlength=p_n + 1)
    cand_ids = [c for c in range(1, p_n + 1) if p_sizes[c] >= MIN_VOX]
    small_ids = [c for c in range(1, p_n + 1) if p_sizes[c] < MIN_VOX]
    pred = describe(p_lab, p_n, p_sizes, cand_ids)
    small = describe(p_lab, p_n, p_sizes, small_ids)

    low = (P > T_LOW) & ~m0
    rec = {}
    if low.any():
        l_lab, l_n = ndi.label(low)
        l_sizes = np.bincount(l_lab.ravel(), minlength=l_n + 1)
        touch_m0 = (np.bincount(l_lab[ndi.binary_dilation(m0)].ravel(),
                                minlength=l_n + 1) if m0.any()
                    else np.zeros(l_n + 1, dtype=np.int64))
        ids = [c for c in range(1, l_n + 1)
               if l_sizes[c] >= MIN_VOX and touch_m0[c] == 0]
        rec = describe(l_lab, l_n, l_sizes, ids)
    return {"case": case, "vml": vml, "gt_vox": int(gt.sum()),
            "gt_sizes": {i: int(g_sizes[i]) for i in range(1, g_n + 1)},
            "pred": pred, "small": small, "rec": rec,
            "gt_positive": bool(gt.any())}


def score_case(bk: dict, keep_pred: set, add_rec: set) -> dict:
    vml = bk["vml"]
    comps = [bk["pred"][c] for c in keep_pred] + [bk["rec"][c] for c in add_rec] \
        + list(bk["small"].values())
    pv = sum(c["size"] for c in comps)
    inter = sum(c["inter"] for c in comps)
    fp = sum(c["size"] for c in comps if c["inter"] == 0) * vml
    hit = set().union(*[c["gt"] for c in comps]) if comps else set()
    fn = sum(s for i, s in bk["gt_sizes"].items() if i not in hit) * vml
    gtv = bk["gt_vox"]
    dice = (2 * inter / (pv + gtv)) if (pv + gtv) else float("nan")
    return {"dice": dice, "fp_ml": fp, "fn_ml": fn, "pred_ml": pv * vml,
            "tp_ml": inter * vml, "gt_positive": bk["gt_positive"],
            "any_fp": fp > 0}


def summarise(books, decisions) -> dict:
    rows = [score_case(bk, *decisions[bk["case"]]) for bk in books]
    pos = [r for r in rows if r["gt_positive"]]
    neg = [r for r in rows if not r["gt_positive"]]
    return {"dice": float(np.mean([r["dice"] for r in pos])),
            "fp_ml": float(np.mean([r["fp_ml"] for r in rows])),
            "fn_ml": float(np.mean([r["fn_ml"] for r in rows])),
            "tp_ml": float(np.sum([r["tp_ml"] for r in pos])),
            "neg_with_fp": int(sum(r["any_fp"] for r in neg)), "n_neg": len(neg)}


def decide(books, pfp, prec, tau_a, tau_b, reflect=False):
    out = {}
    for bk in books:
        c = bk["case"]
        keep = {i for i in bk["pred"] if pfp[(c, i)] <= tau_a}
        add = {i for i in bk["rec"] if prec[(c, i)] >= tau_b}
        if reflect:
            base = score_case(bk, set(bk["pred"]), set())
            new = score_case(bk, keep, add)
            def q(ks, ad):
                comps = [(bk["pred"][i]["size"], 1 - pfp[(c, i)]) for i in ks] + \
                        [(bk["rec"][i]["size"], prec[(c, i)]) for i in ad]
                tot = sum(s for s, _ in comps)
                return (sum(s * p for s, p in comps) / tot) if tot else 1.0
            if q(keep, add) <= q(set(bk["pred"]), set()):
                keep, add = set(bk["pred"]), set()
        out[c] = (keep, add)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--test-cand", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\exp4_refine.json"))
    args = ap.parse_args()

    # --- fit both detectors on val -----------------------------------
    models = {}
    for kind in ("pred", "recover"):
        rows_v, yv, gv, _ = load(args.val_cand, kind)
        feats = GROUPS["d all"]
        Xv, med, _ = matrix(rows_v, feats, None)
        mu, sd = Xv.mean(0), Xv.std(0); sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=5000, C=1.0).fit((Xv - mu) / sd, yv)
        models[kind] = (clf, mu, sd, med, feats)
        print(f"fitted '{kind}' detector on {len(yv)} val candidates "
              f"({int(yv.sum())} positive)")

    def scores(path: Path):
        out = {}
        for kind in ("pred", "recover"):
            clf, mu, sd, med, feats = models[kind]
            rows, y, g, _ = load(path, kind)
            X, _, _ = matrix(rows, feats, med)
            p = clf.predict_proba((X - mu) / sd)[:, 1]
            out[kind] = {(r["case"], int(r["component_id"])): float(v)
                         for r, v in zip(rows, p)}
        return out["pred"], out["recover"]

    sv_fp, sv_rec = scores(args.val_cand)
    st_fp, st_rec = scores(args.test_cand)

    # Every case in the split, not just those with candidates. A case where M0
    # predicts nothing and has nothing above t_low contributes no CSV row, and
    # taking the case list from the CSV silently dropped 6 val and 10 test cases
    # -- shifting every mean and the tumour-free denominator (80 instead of 90).
    # Those cases are part of the result: they are the ones already correct.
    val_cases = sorted(f.stem for f in (UNC / "val").glob("*.npz"))
    test_cases = sorted(f.stem for f in (UNC / "test").glob("*.npz"))
    print(f"\nbuilding bookkeeping: {len(val_cases)} val, {len(test_cases)} test cases")
    books_v = [bookkeep(c, "val") for c in val_cases]
    books_t = [bookkeep(c, "test") for c in test_cases]

    base_v = summarise(books_v, {bk["case"]: (set(bk["pred"]), set()) for bk in books_v})
    print(f"val M0: Dice {base_v['dice']:.4f}  FP {base_v['fp_ml']:.2f} mL  "
          f"neg-with-FP {base_v['neg_with_fp']}/{base_v['n_neg']}  TP vol {base_v['tp_ml']:.0f} mL")

    # --- tune on val --------------------------------------------------
    grid_a = [1.01] + list(np.round(np.arange(0.95, 0.19, -0.05), 2))
    grid_b = [1.01] + list(np.round(np.arange(0.9, 0.09, -0.05), 2))
    results = []
    for ta in grid_a:
        for tb in grid_b:
            d = decide(books_v, sv_fp, sv_rec, ta, tb)
            s = summarise(books_v, d)
            s.update(tau_a=float(ta), tau_b=float(tb))
            results.append(s)
    best_dice = max(results, key=lambda s: s["dice"])
    tp_floor = base_v["tp_ml"] * 0.99
    ok = [s for s in results if s["tp_ml"] >= tp_floor]
    best_cons = min(ok, key=lambda s: (s["neg_with_fp"], -s["dice"])) if ok else best_dice
    print(f"\ntuned on val:")
    print(f"  dice point        tau_A {best_dice['tau_a']:.2f} tau_B {best_dice['tau_b']:.2f}"
          f" -> Dice {best_dice['dice']:.4f} FP {best_dice['fp_ml']:.2f} "
          f"neg-with-FP {best_dice['neg_with_fp']}")
    print(f"  conservative point tau_A {best_cons['tau_a']:.2f} tau_B {best_cons['tau_b']:.2f}"
          f" -> Dice {best_cons['dice']:.4f} FP {best_cons['fp_ml']:.2f} "
          f"neg-with-FP {best_cons['neg_with_fp']} (TP vol kept "
          f"{100*best_cons['tp_ml']/base_v['tp_ml']:.2f}%)")

    out = {"val_m0": base_v, "tuned": {"dice": best_dice, "conservative": best_cons}}

    # --- apply to test ------------------------------------------------
    print(f"\n{'variant':34s} {'Dice':>7s} {'FP mL':>8s} {'FN mL':>8s} {'neg+FP':>8s}")
    print("-" * 70)
    variants = {}
    m0_t = {bk["case"]: (set(bk["pred"]), set()) for bk in books_t}
    variants["M0"] = m0_t
    for pname, pt in (("dice", best_dice), ("conservative", best_cons)):
        ta, tb = pt["tau_a"], pt["tau_b"]
        variants[f"A remove [{pname}]"] = decide(books_t, st_fp, st_rec, ta, 1.01)
        variants[f"B recover [{pname}]"] = decide(books_t, st_fp, st_rec, 1.01, tb)
        variants[f"A+B [{pname}]"] = decide(books_t, st_fp, st_rec, ta, tb)
        variants[f"A+B+reflect [{pname}]"] = decide(books_t, st_fp, st_rec, ta, tb, True)
    summ = {}
    for name, d in variants.items():
        s = summarise(books_t, d)
        summ[name] = s
        print(f"{name:34s} {s['dice']:7.4f} {s['fp_ml']:8.2f} {s['fn_ml']:8.2f} "
              f"{s['neg_with_fp']:>5d}/{s['n_neg']}")
    out["test"] = summ
    out["decisions"] = {name: {c: [sorted(k), sorted(a)] for c, (k, a) in d.items()}
                        for name, d in variants.items()}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
