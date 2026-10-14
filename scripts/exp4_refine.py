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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import GROUPS, load, matrix, patient_of   # noqa: E402

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
            "any_fp": fp > 0,
            # lesion-level sensitivity, the quantity a clinical operating point
            # has to hold: how many distinct true lesions are still touched
            "n_gt": len(bk["gt_sizes"]), "n_hit": len(hit)}


def summarise(books, decisions) -> dict:
    rows = [score_case(bk, *decisions[bk["case"]]) for bk in books]
    pos = [r for r in rows if r["gt_positive"]]
    neg = [r for r in rows if not r["gt_positive"]]
    n_gt = sum(r["n_gt"] for r in pos)
    n_hit = sum(r["n_hit"] for r in pos)
    return {"dice": float(np.mean([r["dice"] for r in pos])),
            "fp_ml": float(np.mean([r["fp_ml"] for r in rows])),
            "fn_ml": float(np.mean([r["fn_ml"] for r in rows])),
            "tp_ml": float(np.sum([r["tp_ml"] for r in pos])),
            "lesion_sens": (n_hit / n_gt) if n_gt else float("nan"),
            "n_gt": n_gt, "n_hit": n_hit,
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


def load_books(split: str, cases: list[str], cache: Path | None) -> list[dict]:
    """Bookkeeping for a split, cached because it is the expensive part.

    Each case costs a map load plus a ground-truth load; the out-of-fold split
    is 738 of them, and both detectors need exactly the same bookkeeping. The
    cache is keyed by split and validated against the case list, so a stale
    file is detected rather than silently reused.
    """
    import pickle
    if cache and cache.exists():
        with open(cache, "rb") as f:
            got = pickle.load(f)
        if [b["case"] for b in got] == cases:
            print(f"  {split}: {len(got)} cases from cache")
            return got
        print(f"  {split}: cache case list differs, rebuilding")
    books = []
    for i, c in enumerate(cases, 1):
        books.append(bookkeep(c, split))
        if i % 100 == 0 or i == len(cases):
            print(f"  {split}: {i}/{len(cases)}", flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(books, f)
    return books


def sweep(books, pfp, prec, grid_a, grid_b):
    out = []
    for ta in grid_a:
        for tb in grid_b:
            s = summarise(books, decide(books, pfp, prec, ta, tb))
            s.update(tau_a=float(ta), tau_b=float(tb))
            out.append(s)
    return out


def pick(results, base, point: str):
    """The three operating points, each a different definition of better."""
    if point == "dice":
        return max(results, key=lambda s: s["dice"])
    if point == "conservative":
        ok = [s for s in results if s["tp_ml"] >= base["tp_ml"] * 0.99]
        return (min(ok, key=lambda s: (s["neg_with_fp"], -s["dice"])) if ok
                else max(results, key=lambda s: s["dice"]))
    if point == "sensitivity":
        ok = [s for s in results if s["lesion_sens"] >= base["lesion_sens"] - 0.01]
        return (min(ok, key=lambda s: (s["fp_ml"], -s["dice"])) if ok
                else max(results, key=lambda s: s["dice"]))
    raise ValueError(point)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--test-cand", type=Path, required=True)
    ap.add_argument("--fit-cand", type=Path, default=None,
                    help="candidates to fit detectors and tune thresholds on; "
                         "without it both happen on val, as in Phase III")
    ap.add_argument("--fit-split", default="oof",
                    help="subdirectory of --maps-root holding the fitting maps")
    ap.add_argument("--detector", choices=("logistic", "gbt"), default="logistic")
    ap.add_argument("--gbt-from", type=Path, default=None,
                    help="exp3 JSON to read out-of-fold-tuned GBT parameters from")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--maps-root", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(r"D:\data\reports\bookcache"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\exp4_refine.json"))
    args = ap.parse_args()
    if args.maps_root:
        globals()["UNC"] = args.maps_root
    feats = [f for f in GROUPS["d all"] if f not in args.drop]
    feats += [f for f in args.add if f not in feats]
    print(f"detector {args.detector!r} on {len(feats)} features")

    gbt_params = {}
    if args.detector == "gbt":
        if not args.gbt_from:
            raise SystemExit("--detector gbt needs --gbt-from")
        j = json.loads(args.gbt_from.read_text())
        gbt_params = {k: j[k]["d all (GBT)"]["params"] for k in ("pred", "recover")}
        print(f"  GBT params from {args.gbt_from.name}: {gbt_params}")

    fit_cand = args.fit_cand or args.val_cand
    fit_split = args.fit_split if args.fit_cand else "val"

    # --- detectors, fitted on the fitting split only ------------------
    models = {}
    for kind in ("pred", "recover"):
        rows_v, yv, gv, _ = load(fit_cand, kind)
        Xv, med, nmiss = matrix(rows_v, feats, None)
        mu, sd = Xv.mean(0), Xv.std(0)
        sd[sd == 0] = 1.0
        if args.detector == "gbt":
            clf = HistGradientBoostingClassifier(random_state=0, early_stopping=False,
                                                 **gbt_params[kind]).fit(Xv, yv)
            std = False
        else:
            clf = LogisticRegression(max_iter=5000, C=1.0).fit((Xv - mu) / sd, yv)
            std = True
        models[kind] = (clf, mu, sd, med, std)
        print(f"fitted {kind!r} on {len(yv)} {fit_split} candidates "
              f"({int(yv.sum())} positive, {nmiss} imputed cells)")

    def scores(path: Path):
        out = {}
        for kind in ("pred", "recover"):
            clf, mu, sd, med, std = models[kind]
            rows, _, _, _ = load(path, kind)
            X, _, _ = matrix(rows, feats, med)
            p = clf.predict_proba((X - mu) / sd if std else X)[:, 1]
            out[kind] = {(r["case"], int(r["component_id"])): float(v)
                         for r, v in zip(rows, p)}
        return out["pred"], out["recover"]

    sf_fp, sf_rec = scores(fit_cand)
    sv_fp, sv_rec = scores(args.val_cand)
    st_fp, st_rec = scores(args.test_cand)

    # Every case in the split, not just those with candidates: a case where M0
    # predicts nothing contributes no CSV row but is still part of the result.
    print("\nbookkeeping")
    splits = {}
    for name in (fit_split, "val", "test"):
        cases = sorted(f.stem for f in (UNC / name).glob("*.npz"))
        splits[name] = load_books(name, cases, args.cache_dir / f"{name}.pkl")

    books_f, books_v, books_t = splits[fit_split], splits["val"], splits["test"]
    base_f = summarise(books_f, {b["case"]: (set(b["pred"]), set()) for b in books_f})
    base_v = summarise(books_v, {b["case"]: (set(b["pred"]), set()) for b in books_v})
    base_t = summarise(books_t, {b["case"]: (set(b["pred"]), set()) for b in books_t})
    for nm, b in ((fit_split, base_f), ("val", base_v), ("test", base_t)):
        print(f"{nm:5s} M0: Dice {b['dice']:.4f}  FP {b['fp_ml']:6.2f} mL  "
              f"neg-with-FP {b['neg_with_fp']:>3d}/{b['n_neg']:<3d} "
              f"lesion sens {b['lesion_sens']:.4f}")

    # --- thresholds, tuned on the fitting split -----------------------
    grid_a = [1.01] + list(np.round(np.arange(0.95, 0.19, -0.05), 2))
    grid_b = [1.01] + list(np.round(np.arange(0.9, 0.09, -0.05), 2))
    results = sweep(books_f, sf_fp, sf_rec, grid_a, grid_b)
    points = ("dice", "conservative", "sensitivity")
    chosen = {p: pick(results, base_f, p) for p in points}

    # Patient-grouped CV *within* the fitting split, to say whether the chosen
    # threshold is a stable property of the data or an artefact of one draw.
    groups = np.array([patient_of(b["case"]) for b in books_f])
    gkf = GroupKFold(n_splits=5)
    stab = {p: [] for p in points}
    for tr, _ in gkf.split(np.zeros(len(books_f)), None, groups):
        sub = [books_f[i] for i in tr]
        b0 = summarise(sub, {b["case"]: (set(b["pred"]), set()) for b in sub})
        rs = sweep(sub, sf_fp, sf_rec, grid_a, grid_b)
        for p in points:
            s = pick(rs, b0, p)
            stab[p].append((s["tau_a"], s["tau_b"]))
    print(f"\nthresholds tuned on {fit_split} "
          f"({len(books_f)} cases, {len(set(groups))} patients)")
    out = {"detector": args.detector, "features": feats, "fit_split": fit_split,
           "gbt_params": gbt_params,
           "m0": {fit_split: base_f, "val": base_v, "test": base_t}, "tuned": {}}
    for p in points:
        s = chosen[p]
        ta = [t[0] for t in stab[p]]
        tb = [t[1] for t in stab[p]]
        print(f"  {p:13s} tau_A {s['tau_a']:.2f} tau_B {s['tau_b']:.2f}"
              f"  | across 5 patient-grouped folds tau_A "
              f"{min(ta):.2f}-{max(ta):.2f}, tau_B {min(tb):.2f}-{max(tb):.2f}")
        out["tuned"][p] = {**s, "cv_tau_a": ta, "cv_tau_b": tb}

    # --- val as a check, then test once -------------------------------
    for split_name, books, sfp, srec in (("val", books_v, sv_fp, sv_rec),
                                         ("test", books_t, st_fp, st_rec)):
        print(f"\n{split_name.upper()}  {'variant':30s} {'Dice':>7s} {'FP mL':>8s} "
              f"{'FN mL':>8s} {'neg+FP':>9s} {'lesSens':>8s}")
        print("-" * 82)
        base = base_v if split_name == "val" else base_t
        vs = {"M0": {b["case"]: (set(b["pred"]), set()) for b in books}}
        for p in points:
            ta, tb = chosen[p]["tau_a"], chosen[p]["tau_b"]
            vs[f"A remove [{p}]"] = decide(books, sfp, srec, ta, 1.01)
            vs[f"B recover [{p}]"] = decide(books, sfp, srec, 1.01, tb)
            vs[f"A+B [{p}]"] = decide(books, sfp, srec, ta, tb)
            vs[f"A+B+reflect [{p}]"] = decide(books, sfp, srec, ta, tb, True)
        summ = {}
        for name, d in vs.items():
            s = summarise(books, d)
            summ[name] = s
            flag = ""
            if name != "M0":
                ds = s["lesion_sens"] - base["lesion_sens"]
                flag = f"  dSens {ds:+.4f}" + ("  OVER BUDGET" if ds < -0.01 else "")
            print(f"      {name:30s} {s['dice']:7.4f} {s['fp_ml']:8.2f} "
                  f"{s['fn_ml']:8.2f} {s['neg_with_fp']:>5d}/{s['n_neg']:<3d} "
                  f"{s['lesion_sens']:8.4f}{flag}")
        out[split_name] = summ
        if split_name == "test":
            out["decisions"] = {n: {c: [sorted(k), sorted(a)] for c, (k, a) in d.items()}
                                for n, d in vs.items()}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
