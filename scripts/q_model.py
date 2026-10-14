"""An independent case-quality model Q, and the reflection gate rebuilt on it.

The reflection variant in `exp4_refine.py` scores a case with a volume-weighted
mean of the *action detector's own* probabilities, so "accept the edit only if
Q improves" reduces to "accept only if the detector agrees with itself". This
builds Q from scratch so that the gate carries information the detector does
not already have:

  different target     Q regresses the case's Dice, not a component's label.
  different unit       Q sees one whole mask per case, not one component.
  detector-free        no detector score is an input, and the training masks
                       are drawn at random rather than produced by thresholding
                       the detector, so Q never sees its decision boundary.
  honest predictions   on out-of-fold cases Q is predicted by the patient-
                       grouped fold model that did not train on that patient.

Training pairs are (mask, Dice). For each case we take M0 plus `--n-draw`
random masks, each keeping a pred component with probability u ~ U(0.5, 1) and
adding a recovery component with probability v ~ U(0, 0.35), u and v drawn once
per mask. Dice is exact, from the same bookkeeping Experiment 4 uses, with the
convention that an empty prediction on a tumour-free case scores 1.0 -- that is
the case the whole removal action exists to produce, so it must not be nan.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp4_refine as E
from exp4_refine import load_books, score_case, summarise, patient_of
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

# Per-component columns Q aggregates. No detector score, and no column the
# detector uses as a *component-level* decision variable is used that way here;
# everything is pooled to one number per mask.
WMEAN = ["p_mean", "uent_mean", "ustab_mean", "suv_mean", "suv_max_over_liver",
         "frac_physio", "frac_liver", "dist_physio_mm", "sphericity",
         "elongation", "hu_mean", "edge_body_mm"]
QFEATS = (["q_vol_ml", "q_ncomp", "q_logvol_max", "q_logvol_min", "q_suv_max"]
          + [f"q_{c}" for c in WMEAN]
          + ["c_npred", "c_nrec", "c_liver_suv", "c_vol_m0_ml"])


def read_cand(path: Path) -> dict:
    """{(case, kind, id): row} with floats, plus per-case context."""
    out, ctx = {}, {}
    for r in csv.DictReader(open(path)):
        key = (r["case"], r["kind"], int(r["component_id"]))
        d = {}
        for c in WMEAN + ["suv_max", "volume_ml", "liver_suv_mean"]:
            v = r.get(c, "")
            d[c] = float(v) if v not in ("", "nan", "NaN") else np.nan
        out[key] = d
        c0 = ctx.setdefault(r["case"], {"c_npred": 0, "c_nrec": 0,
                                        "c_liver_suv": np.nan})
        c0["c_npred" if r["kind"] == "pred" else "c_nrec"] += 1
        if np.isfinite(d["liver_suv_mean"]):
            c0["c_liver_suv"] = d["liver_suv_mean"]
    return out, ctx


def mask_features(bk, cand, ctx, keep, add) -> np.ndarray:
    c = bk["case"]
    comps = [(bk["pred"][i]["size"], cand.get((c, "pred", i))) for i in keep] + \
            [(bk["rec"][i]["size"], cand.get((c, "recover", i))) for i in add]
    comps = [(s, d) for s, d in comps if d is not None]
    vml = bk["vml"]
    f = {k: np.nan for k in QFEATS}
    tot = sum(s for s, _ in comps)
    f["q_vol_ml"] = tot * vml
    f["q_ncomp"] = len(comps)
    if comps:
        sizes = np.array([s for s, _ in comps], float)
        f["q_logvol_max"] = float(np.log10(sizes.max() * vml + 1e-6))
        f["q_logvol_min"] = float(np.log10(sizes.min() * vml + 1e-6))
        sm = [d["suv_max"] for _, d in comps if np.isfinite(d["suv_max"])]
        f["q_suv_max"] = float(max(sm)) if sm else np.nan
        for col in WMEAN:
            v = np.array([d[col] for _, d in comps], float)
            ok = np.isfinite(v)
            f[f"q_{col}"] = (float(np.sum(sizes[ok] * v[ok]) / sizes[ok].sum())
                             if ok.any() else np.nan)
    else:                      # an entirely emptied mask is a real state
        f["q_logvol_max"] = f["q_logvol_min"] = -6.0
        f["q_suv_max"] = 0.0
        for col in WMEAN:
            f[f"q_{col}"] = 0.0
    k = ctx.get(c, {})
    f["c_npred"] = k.get("c_npred", 0)
    f["c_nrec"] = k.get("c_nrec", 0)
    f["c_liver_suv"] = k.get("c_liver_suv", np.nan)
    f["c_vol_m0_ml"] = sum(d["size"] for d in bk["pred"].values()) * vml
    return np.array([f[k2] for k2 in QFEATS], float)


def dice_of(bk, keep, add) -> float:
    s = score_case(bk, keep, add)
    d = s["dice"]
    # both empty: a tumour-free case left clean is a perfect result, not nan
    return 1.0 if not np.isfinite(d) else d


def build_pairs(books, cand, ctx, n_draw, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for bk in books:
        pred, rec = list(bk["pred"]), list(bk["rec"])
        masks = [(set(pred), set())]
        for _ in range(n_draw):
            u, v = rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.35)
            masks.append(({i for i in pred if rng.random() < u},
                          {i for i in rec if rng.random() < v}))
        for keep, add in masks:
            X.append(mask_features(bk, cand, ctx, keep, add))
            y.append(dice_of(bk, keep, add))
            g.append(patient_of(bk["case"]))
    return np.array(X, float), np.array(y, float), np.array(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-cand", type=Path, required=True)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--test-cand", type=Path, required=True)
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\reports\bookcache"))
    ap.add_argument("--n-draw", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\q_model.json"))
    args = ap.parse_args()

    E.UNC = args.maps_root
    books = {}
    for name in ("oof", "val", "test"):
        cases = sorted(f.stem for f in (args.maps_root / name).glob("*.npz"))
        books[name] = load_books(name, cases, args.cache_dir / f"{name}.pkl")
    cand = {"oof": read_cand(args.fit_cand), "val": read_cand(args.val_cand),
            "test": read_cand(args.test_cand)}

    print(f"\nbuilding training pairs: 738 cases x (1 + {args.n_draw}) masks")
    X, y, g = build_pairs(books["oof"], *cand["oof"], args.n_draw)
    print(f"  {X.shape[0]} pairs, {X.shape[1]} features, "
          f"Dice {y.min():.3f}-{y.max():.3f} mean {y.mean():.3f}")

    # honest CV: a patient's masks are scored by a model that never saw them
    gkf = GroupKFold(n_splits=5)
    pred = np.full(len(y), np.nan)
    fold_models = []
    for tr, te in gkf.split(X, y, g):
        m = HistGradientBoostingRegressor(random_state=0, max_depth=3,
                                          learning_rate=0.06, max_iter=400)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        fold_models.append((set(g[te]), m))
    ok = np.isfinite(pred)
    r2 = 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    rho = spearmanr(y[ok], pred[ok]).statistic
    print(f"\nQ, patient-grouped CV over out-of-fold masks:")
    print(f"  R2 {r2:.4f}   Spearman {rho:.4f}   n {ok.sum()}")
    # split by case polarity, because the two regimes are very different
    posmask = np.array([bk["gt_positive"] for bk in books["oof"]
                        for _ in range(args.n_draw + 1)])
    for nm, sel in (("tumour-positive cases", posmask), ("tumour-free cases", ~posmask)):
        s = sel & ok
        if s.sum() > 10:
            r2s = 1 - np.sum((y[s] - pred[s]) ** 2) / np.sum((y[s] - y[s].mean()) ** 2)
            print(f"  {nm:22s} R2 {r2s:+.4f}  Spearman "
                  f"{spearmanr(y[s], pred[s]).statistic:.4f}  n {int(s.sum())}")

    # the ranking that actually matters: within a case, does Q order masks right?
    per_case = []
    k = args.n_draw + 1
    for i, bk in enumerate(books["oof"]):
        yy, pp = y[i * k:(i + 1) * k], pred[i * k:(i + 1) * k]
        if np.isfinite(pp).all() and yy.std() > 1e-9:
            per_case.append(spearmanr(yy, pp).statistic)
    per_case = np.array([v for v in per_case if np.isfinite(v)])
    print(f"  within-case mask ranking: median Spearman {np.median(per_case):.4f} "
          f"over {len(per_case)} cases")

    full = HistGradientBoostingRegressor(random_state=0, max_depth=3,
                                         learning_rate=0.06, max_iter=400).fit(X, y)
    out = {"features": QFEATS, "n_pairs": int(X.shape[0]), "r2_cv": float(r2),
           "spearman_cv": float(rho),
           "within_case_spearman_median": float(np.median(per_case))}

    def qfun(split):
        """Q for a split: fold model for out-of-fold cases, full model elsewhere."""
        if split != "oof":
            return lambda bk, keep, add: float(full.predict(
                mask_features(bk, *cand[split], keep, add)[None])[0])

        def f(bk, keep, add):
            p = patient_of(bk["case"])
            for grp, m in fold_models:
                if p in grp:
                    return float(m.predict(
                        mask_features(bk, *cand["oof"], keep, add)[None])[0])
            return float(full.predict(
                mask_features(bk, *cand["oof"], keep, add)[None])[0])
        return f
    out["qfun_ready"] = True
    import pickle
    with open(args.out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"full": full, "folds": fold_models, "feats": QFEATS}, f)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {args.out} and {args.out.with_suffix('.pkl')}")


if __name__ == "__main__":
    main()
