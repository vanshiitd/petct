"""Reporting pass over Experiment 4: ensemble vs single M0, a selection rule
fixed on out-of-fold/val, paired-bootstrap CIs on test, and FROC curves.

Nothing here tunes anything. The thresholds come from `exp4_refine.py`, which
tuned them on the out-of-fold split; this script only chooses *which already
tuned variant* to report, and it chooses with a rule evaluated on out-of-fold
and val. Test is read for the final numbers and for the curves, never to pick.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp4_refine as E
from exp4_refine import (load, matrix, GROUPS, decide, summarise, score_case,
                         load_books, sweep, pick, patient_of)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

POINTS = ("dice", "conservative", "sensitivity")
ACTIONS = ("A remove", "B recover", "A+B", "A+B+reflect")
BUDGET = -0.01          # lesion sensitivity may fall at most 1 point
NBOOT = 2000


# --------------------------------------------------------------- per-case ---
def per_case(books, decisions) -> list[dict]:
    """score_case for every book, plus the FP lesion count FROC needs."""
    rows = []
    for bk in books:
        keep, add = decisions[bk["case"]]
        r = score_case(bk, keep, add)
        comps = [bk["pred"][c] for c in keep] + [bk["rec"][c] for c in add] \
            + list(bk["small"].values())
        r["n_fp_comp"] = sum(1 for c in comps if c["inter"] == 0)
        r["case"] = bk["case"]
        rows.append(r)
    return rows


def agg(rows, idx=None) -> dict:
    """Aggregate per-case rows, optionally over a bootstrap index."""
    sel = rows if idx is None else [rows[i] for i in idx]
    pos = [r for r in sel if r["gt_positive"]]
    n_gt = sum(r["n_gt"] for r in pos)
    n_hit = sum(r["n_hit"] for r in pos)
    return {"dice": float(np.mean([r["dice"] for r in pos])) if pos else float("nan"),
            "fp_ml": float(np.mean([r["fp_ml"] for r in sel])),
            "fn_ml": float(np.mean([r["fn_ml"] for r in sel])),
            "lesion_sens": (n_hit / n_gt) if n_gt else float("nan"),
            "fp_per_patient": float(np.mean([r["n_fp_comp"] for r in sel])),
            "neg_with_fp": int(sum(r["any_fp"] for r in sel if not r["gt_positive"])),
            "n_neg": sum(1 for r in sel if not r["gt_positive"])}


def paired_boot(rows_a, rows_b, keys, seed=0, nboot=NBOOT):
    """Paired case-level bootstrap of (a - b). Test is 200 patients, one scan
    each, so resampling cases is resampling patients."""
    assert [r["case"] for r in rows_a] == [r["case"] for r in rows_b]
    n = len(rows_a)
    rng = np.random.default_rng(seed)
    obs = {k: agg(rows_a)[k] - agg(rows_b)[k] for k in keys}
    draws = {k: [] for k in keys}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        ga, gb = agg(rows_a, idx), agg(rows_b, idx)
        for k in keys:
            draws[k].append(ga[k] - gb[k])
    out = {}
    for k in keys:
        d = np.asarray(draws[k], float)
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"delta": obs[k], "lo": float(lo), "hi": float(hi),
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def fmt(d) -> str:
    return f"{d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" + \
        ("" if d["excludes_zero"] else "  (straddles 0)")


# ------------------------------------------------------------- detectors ---
def fit_detectors(fit_cand, feats, kind_params, detector):
    models = {}
    for kind in ("pred", "recover"):
        rows_v, yv, _, _ = load(fit_cand, kind)
        Xv, med, _ = matrix(rows_v, feats, None)
        mu, sd = Xv.mean(0), Xv.std(0)
        sd[sd == 0] = 1.0
        if detector == "gbt":
            clf = HistGradientBoostingClassifier(
                random_state=0, early_stopping=False, **kind_params[kind]).fit(Xv, yv)
            std = False
        else:
            clf = LogisticRegression(max_iter=5000, C=1.0).fit((Xv - mu) / sd, yv)
            std = True
        models[kind] = (clf, mu, sd, med, std)
    return models


def score_split(models, path, feats):
    out = {}
    for kind in ("pred", "recover"):
        clf, mu, sd, med, std = models[kind]
        rows, _, _, _ = load(path, kind)
        X, _, _ = matrix(rows, feats, med)
        p = clf.predict_proba((X - mu) / sd if std else X)[:, 1]
        out[kind] = {(r["case"], int(r["component_id"])): float(v)
                     for r, v in zip(rows, p)}
    return out["pred"], out["recover"]


def variants(books, sfp, srec, taus):
    """The 12 tuned variants plus M0, as decision dicts."""
    vs = {"M0": {b["case"]: (set(b["pred"]), set()) for b in books}}
    for p in POINTS:
        ta, tb = taus[p]
        vs[f"A remove [{p}]"] = decide(books, sfp, srec, ta, 1.01)
        vs[f"B recover [{p}]"] = decide(books, sfp, srec, 1.01, tb)
        vs[f"A+B [{p}]"] = decide(books, sfp, srec, ta, tb)
        vs[f"A+B+reflect [{p}]"] = decide(books, sfp, srec, ta, tb, True)
    return vs


# ------------------------------------------------------------------ FROC ---
def froc(books, sfp, srec, mode, taus_fixed=None):
    """(FP lesions per patient, lesion sensitivity) as a threshold sweeps.

    mode 'A' sweeps tau_A with recovery off, 'B' sweeps tau_B with removal off,
    'AB' takes the Pareto front over the full 2-D grid so the combined action
    is shown at its best rather than along one arbitrary line through it.
    """
    ga = [1.01] + list(np.round(np.arange(0.95, 0.04, -0.05), 2))
    gb = [1.01] + list(np.round(np.arange(0.95, 0.04, -0.05), 2))
    pts = []
    if mode == "A":
        grid = [(t, 1.01) for t in ga]
    elif mode == "B":
        grid = [(1.01, t) for t in gb]
    else:
        grid = [(a, b) for a in ga for b in gb]
    for ta, tb in grid:
        s = agg(per_case(books, decide(books, sfp, srec, ta, tb)))
        pts.append((s["fp_per_patient"], s["lesion_sens"], ta, tb))
    if mode == "AB":                       # upper-left Pareto front
        pts.sort(key=lambda p: (p[0], -p[1]))
        front, best = [], -1.0
        for p in pts:
            if p[1] > best:
                front.append(p)
                best = p[1]
        pts = front
    else:
        pts.sort(key=lambda p: p[0])
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-cand", type=Path, required=True)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--test-cand", type=Path, required=True)
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--single-root", type=Path,
                    default=Path(r"D:\data\uncertainty_m0L"),
                    help="maps of the single Dataset505 model, for the M0 comparison")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--gbt-from", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\reports\bookcache"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\exp4_report.json"))
    ap.add_argument("--fig", type=Path, default=Path(r"D:\data\reports\fig_froc.png"))
    args = ap.parse_args()

    feats = [f for f in GROUPS["d all"] if f not in args.drop]
    feats += [f for f in args.add if f not in feats]
    report = {"features": feats}

    # ---- books ------------------------------------------------------
    E.UNC = args.maps_root
    books = {}
    for name in ("oof", "val", "test"):
        cases = sorted(f.stem for f in (args.maps_root / name).glob("*.npz"))
        books[name] = load_books(name, cases, args.cache_dir / f"{name}.pkl")

    # ---- 1. ensemble M0 vs the single Dataset505 model, on test ------
    print("\n" + "=" * 78)
    print("1. M0: 5-fold ensemble vs single Dataset505 model (test, n=200)")
    E.UNC = args.single_root
    cases_t = [b["case"] for b in books["test"]]
    books_single = load_books("test", cases_t, args.cache_dir / "test_single505.pkl")
    E.UNC = args.maps_root
    rows_ens = per_case(books["test"],
                        {b["case"]: (set(b["pred"]), set()) for b in books["test"]})
    rows_sgl = per_case(books_single,
                        {b["case"]: (set(b["pred"]), set()) for b in books_single})
    a_ens, a_sgl = agg(rows_ens), agg(rows_sgl)
    print(f"  {'':14s}{'Dice':>8s}{'FP mL':>9s}{'FN mL':>9s}{'lesSens':>9s}{'FP/pt':>8s}")
    for nm, a in (("ensemble", a_ens), ("single 505", a_sgl)):
        print(f"  {nm:14s}{a['dice']:8.4f}{a['fp_ml']:9.2f}{a['fn_ml']:9.2f}"
              f"{a['lesion_sens']:9.4f}{a['fp_per_patient']:8.2f}")
    bs = paired_boot(rows_ens, rows_sgl,
                     ("dice", "fp_ml", "fn_ml", "lesion_sens", "fp_per_patient"))
    for k, v in bs.items():
        print(f"  ensemble - single  {k:16s} {fmt(v)}")
    report["m0_ensemble_vs_single"] = {"ensemble": a_ens, "single505": a_sgl,
                                       "bootstrap": bs}

    # ---- detectors, both ---------------------------------------------
    gj = json.loads(args.gbt_from.read_text())
    gbt_params = {k: gj[k]["d all (GBT)"]["params"] for k in ("pred", "recover")}
    tuned = {
        "logistic": json.loads(Path(r"D:\data\reports\exp4_cv_anat_logistic.json").read_text()),
        "gbt": json.loads(Path(r"D:\data\reports\exp4_cv_anat_gbt.json").read_text()),
    }

    for det in ("logistic", "gbt"):
        print("\n" + "=" * 78)
        print(f"2. selection rule and test CIs -- {det}")
        models = fit_detectors(args.fit_cand, feats, gbt_params, det)
        sc = {"oof": score_split(models, args.fit_cand, feats),
              "val": score_split(models, args.val_cand, feats),
              "test": score_split(models, args.test_cand, feats)}
        taus = {p: (tuned[det]["tuned"][p]["tau_a"], tuned[det]["tuned"][p]["tau_b"])
                for p in POINTS}
        print(f"  thresholds from out-of-fold tuning: "
              + ", ".join(f"{p} ({a:.2f},{b:.2f})" for p, (a, b) in taus.items()))

        rows = {}
        for split in ("oof", "val", "test"):
            vs = variants(books[split], *sc[split], taus)
            rows[split] = {n: per_case(books[split], d) for n, d in vs.items()}

        # Rule R, fixed on out-of-fold and val only.
        base = {s: agg(rows[s]["M0"]) for s in ("oof", "val", "test")}
        names = [n for n in rows["oof"] if n != "M0"]
        print(f"\n  {'variant':26s}{'dSens oof':>11s}{'dSens val':>11s}"
              f"{'FP mL oof':>11s}{'Dice oof':>10s}  qualifies")
        qual = []
        for n in names:
            o, v = agg(rows["oof"][n]), agg(rows["val"][n])
            do = o["lesion_sens"] - base["oof"]["lesion_sens"]
            dv = v["lesion_sens"] - base["val"]["lesion_sens"]
            ok = do >= BUDGET and dv >= BUDGET
            if ok:
                qual.append((o["fp_ml"], -o["dice"], n))
            print(f"  {n:26s}{do:+11.4f}{dv:+11.4f}{o['fp_ml']:11.2f}"
                  f"{o['dice']:10.4f}  {'yes' if ok else 'no'}")
        report.setdefault("rule", {})[det] = {
            "qualifying": [q[2] for q in sorted(qual)],
            "thresholds": {p: list(t) for p, t in taus.items()}}

        if not qual:
            print("\n  RULE R selects nothing: no variant holds lesion "
                  "sensitivity within 1 point on BOTH out-of-fold and val.")
            report["rule"][det]["selected"] = None
            chosen_name = None
        else:
            chosen_name = sorted(qual)[0][2]
            print(f"\n  RULE R selects: {chosen_name}  "
                  f"(lowest out-of-fold FP volume among {len(qual)} qualifying)")
            report["rule"][det]["selected"] = chosen_name

        # test numbers, with CIs, for the selected variant (and M0 reference)
        print(f"\n  {'split':6s}{'variant':26s}{'Dice':>8s}{'FP mL':>9s}"
              f"{'lesSens':>9s}{'FP/pt':>8s}")
        for split in ("val", "test"):
            a0 = base[split]
            print(f"  {split:6s}{'M0':26s}{a0['dice']:8.4f}{a0['fp_ml']:9.2f}"
                  f"{a0['lesion_sens']:9.4f}{a0['fp_per_patient']:8.2f}")
            if chosen_name:
                a = agg(rows[split][chosen_name])
                print(f"  {split:6s}{chosen_name:26s}{a['dice']:8.4f}{a['fp_ml']:9.2f}"
                      f"{a['lesion_sens']:9.4f}{a['fp_per_patient']:8.2f}")
        if chosen_name:
            bs = paired_boot(rows["test"][chosen_name], rows["test"]["M0"],
                             ("lesion_sens", "fp_ml", "dice"))
            print("\n  test, selected - M0, paired bootstrap over 200 patients:")
            for k, v in bs.items():
                print(f"    {k:14s} {fmt(v)}")
            report["rule"][det]["test_bootstrap"] = bs

        # ---- 3. FROC -------------------------------------------------
        report.setdefault("froc", {})[det] = {}
        for split in ("val", "test"):
            report["froc"][det][split] = {
                "M0": [agg(rows[split]["M0"])["fp_per_patient"],
                       agg(rows[split]["M0"])["lesion_sens"]],
                "A": froc(books[split], *sc[split], "A"),
                "B": froc(books[split], *sc[split], "B"),
                "AB": froc(books[split], *sc[split], "AB")}
            print(f"  FROC {det} {split}: "
                  + ", ".join(f"{m} {len(report['froc'][det][split][m])} pts"
                              for m in ("A", "B", "AB")))

    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten to {args.out}")

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, det in enumerate(("logistic", "gbt")):
        for j, split in enumerate(("val", "test")):
            ax = axes[i][j]
            f = report["froc"][det][split]
            for mode, col, lab in (("A", "tab:red", "A remove"),
                                   ("B", "tab:blue", "B recover"),
                                   ("AB", "tab:green", "A+B (Pareto)")):
                p = np.array([[q[0], q[1]] for q in f[mode]], float)
                ax.plot(p[:, 0], p[:, 1], "-o", ms=2.5, lw=1.2, color=col, label=lab)
            ax.plot(f["M0"][0], f["M0"][1], "k*", ms=14, label="M0", zorder=5)
            ax.axhline(f["M0"][1] - 0.01, color="k", ls=":", lw=1,
                       label="M0 sens - 1 pt")
            ax.set_title(f"{det}, {split}")
            ax.set_xlabel("false-positive lesions per patient")
            ax.set_ylabel("lesion sensitivity")
            ax.grid(alpha=.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("FROC: lesion sensitivity vs false-positive lesions per patient\n"
                 "anatomical feature set, detectors fitted on 738 out-of-fold scans")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=140)
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
