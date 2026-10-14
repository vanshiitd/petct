"""Experiment 4, final pass: the reflection gate on the independent Q model,
Rule R applied again, and the clinical view of what removal actually removes.

Three things the earlier passes could not do:

  reflectQ   the gate accepts a case's edit only if the *independent* case
             quality model from `q_model.py` predicts a higher Dice. Q never
             sees a detector score, and on out-of-fold cases it is evaluated
             with the patient-grouped fold model that did not train on that
             patient, so the gate is not the detector agreeing with itself.

  lesions    removal is reported the way a reader would experience it: how
             many true lesions are lost and how large they were, against how
             many false-positive components disappear and how large those were.

  budgeted   the best false-positive reduction reachable while lesion
             sensitivity stays within one point of M0, computed on each split
             separately so the gap between "tunable" and "transferable" is
             explicit.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp4_refine as E
from exp4_refine import (load, matrix, GROUPS, decide, score_case, load_books,
                         patient_of)
from exp4_report import (per_case, agg, paired_boot, fmt, fit_detectors,
                         score_split, POINTS, BUDGET)
from q_model import mask_features, read_cand
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

SIZE_EDGES = [0.0, 0.5, 1.0, 5.0, 20.0, np.inf]
SIZE_NAMES = ["<0.5 mL", "0.5-1 mL", "1-5 mL", "5-20 mL", ">20 mL"]


def bucket(v: float) -> int:
    return int(np.searchsorted(SIZE_EDGES, v, side="right") - 1)


def q_predictor(qpkl: Path, cand_by_split: dict):
    with open(qpkl, "rb") as f:
        Q = pickle.load(f)

    def make(split):
        def f(bk, keep, add):
            x = mask_features(bk, *cand_by_split[split], keep, add)[None]
            if split == "oof":
                p = patient_of(bk["case"])
                for grp, m in Q["folds"]:
                    if p in grp:
                        return float(m.predict(x)[0])
            return float(Q["full"].predict(x)[0])
        return f
    return make


def decide_q(books, pfp, prec, tau_a, tau_b, qf):
    """A+B, kept only where the independent Q predicts an improvement."""
    out = {}
    for bk in books:
        c = bk["case"]
        keep = {i for i in bk["pred"] if pfp[(c, i)] <= tau_a}
        add = {i for i in bk["rec"] if prec[(c, i)] >= tau_b}
        base = (set(bk["pred"]), set())
        if qf(bk, keep, add) <= qf(bk, *base):
            keep, add = base
        out[c] = (keep, add)
    return out


def hits(bk, keep, add) -> set:
    comps = [bk["pred"][c] for c in keep] + [bk["rec"][c] for c in add] \
        + list(bk["small"].values())
    return set().union(*[c["gt"] for c in comps]) if comps else set()


def lesion_table(books, decisions) -> dict:
    """True lesions lost and false-positive components removed, by size."""
    lost = [0] * len(SIZE_NAMES)
    gained = [0] * len(SIZE_NAMES)
    fp_removed = [0] * len(SIZE_NAMES)
    fp_vol = [0.0] * len(SIZE_NAMES)
    lost_vol = [0.0] * len(SIZE_NAMES)
    for bk in books:
        keep, add = decisions[bk["case"]]
        vml = bk["vml"]
        h0 = hits(bk, set(bk["pred"]), set())
        h1 = hits(bk, keep, add)
        for gid in h0 - h1:
            b = bucket(bk["gt_sizes"][gid] * vml)
            lost[b] += 1
            lost_vol[b] += bk["gt_sizes"][gid] * vml
        for gid in h1 - h0:
            gained[bucket(bk["gt_sizes"][gid] * vml)] += 1
        for cid in set(bk["pred"]) - keep:
            d = bk["pred"][cid]
            if d["inter"] == 0:
                b = bucket(d["size"] * vml)
                fp_removed[b] += 1
                fp_vol[b] += d["size"] * vml
    return {"lesions_lost": lost, "lesions_lost_ml": lost_vol,
            "lesions_gained": gained, "fp_components_removed": fp_removed,
            "fp_volume_removed_ml": fp_vol}


def print_lesion_table(name, t, books):
    n_gt = [0] * len(SIZE_NAMES)
    for bk in books:
        for gid, s in bk["gt_sizes"].items():
            n_gt[bucket(s * bk["vml"])] += 1
    print(f"\n  {name}")
    print(f"    {'size':10s}{'GT lesions':>11s}{'lost':>7s}{'lost%':>8s}"
          f"{'gained':>8s}{'FP comps cut':>14s}{'FP mL cut':>11s}")
    for i, nm in enumerate(SIZE_NAMES):
        pct = 100 * t["lesions_lost"][i] / n_gt[i] if n_gt[i] else 0.0
        print(f"    {nm:10s}{n_gt[i]:11d}{t['lesions_lost'][i]:7d}{pct:7.1f}%"
              f"{t['lesions_gained'][i]:8d}{t['fp_components_removed'][i]:14d}"
              f"{t['fp_volume_removed_ml'][i]:11.1f}")
    print(f"    {'total':10s}{sum(n_gt):11d}{sum(t['lesions_lost']):7d}"
          f"{100*sum(t['lesions_lost'])/max(sum(n_gt),1):7.1f}%"
          f"{sum(t['lesions_gained']):8d}{sum(t['fp_components_removed']):14d}"
          f"{sum(t['fp_volume_removed_ml']):11.1f}")


def budgeted_best(books, sfp, srec, base):
    """Lowest FP volume with lesion sensitivity within 1 point, on this split."""
    best = None
    for ta in [1.01] + list(np.round(np.arange(0.95, 0.04, -0.05), 2)):
        for tb in [1.01] + list(np.round(np.arange(0.95, 0.04, -0.05), 2)):
            s = agg(per_case(books, decide(books, sfp, srec, ta, tb)))
            if s["lesion_sens"] >= base["lesion_sens"] + BUDGET:
                if best is None or s["fp_ml"] < best[0]["fp_ml"]:
                    best = (s, ta, tb)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-cand", type=Path, required=True)
    ap.add_argument("--val-cand", type=Path, required=True)
    ap.add_argument("--test-cand", type=Path, required=True)
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--qpkl", type=Path, default=Path(r"D:\data\reports\q_model.pkl"))
    ap.add_argument("--gbt-from", type=Path, required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\reports\bookcache"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\data\reports\exp4_final.json"))
    args = ap.parse_args()

    feats = [f for f in GROUPS["d all"] if f not in args.drop]
    feats += [f for f in args.add if f not in feats]
    E.UNC = args.maps_root
    books, paths = {}, {"oof": args.fit_cand, "val": args.val_cand,
                        "test": args.test_cand}
    for name in ("oof", "val", "test"):
        cases = sorted(f.stem for f in (args.maps_root / name).glob("*.npz"))
        books[name] = load_books(name, cases, args.cache_dir / f"{name}.pkl")
    cand_by_split = {k: read_cand(v) for k, v in paths.items()}
    make_q = q_predictor(args.qpkl, cand_by_split)
    qf = {s: make_q(s) for s in ("oof", "val", "test")}

    gj = json.loads(args.gbt_from.read_text())
    gbt_params = {k: gj[k]["d all (GBT)"]["params"] for k in ("pred", "recover")}
    tuned = {d: json.loads(Path(rf"D:\data\reports\exp4_cv_anat_{d}.json").read_text())
             for d in ("logistic", "gbt")}

    report = {"features": feats}
    for det in ("logistic", "gbt"):
        print("\n" + "=" * 82)
        print(f"detector: {det}")
        models = fit_detectors(args.fit_cand, feats, gbt_params, det)
        sc = {s: score_split(models, paths[s], feats) for s in paths}
        taus = {p: (tuned[det]["tuned"][p]["tau_a"], tuned[det]["tuned"][p]["tau_b"])
                for p in POINTS}

        rows, dec = {}, {}
        for s in ("oof", "val", "test"):
            d = {"M0": {b["case"]: (set(b["pred"]), set()) for b in books[s]}}
            for p in POINTS:
                ta, tb = taus[p]
                d[f"A remove [{p}]"] = decide(books[s], *sc[s], ta, 1.01)
                d[f"B recover [{p}]"] = decide(books[s], *sc[s], 1.01, tb)
                d[f"A+B [{p}]"] = decide(books[s], *sc[s], ta, tb)
                d[f"A+B+reflectQ [{p}]"] = decide_q(books[s], *sc[s], ta, tb, qf[s])
            dec[s] = d
            rows[s] = {n: per_case(books[s], dd) for n, dd in d.items()}

        base = {s: agg(rows[s]["M0"]) for s in rows}
        names = [n for n in rows["oof"] if n != "M0"]
        print(f"\n  {'variant':28s}{'dSens oof':>11s}{'dSens val':>11s}"
              f"{'FP mL oof':>11s}{'Dice oof':>10s}  qualifies")
        qual = []
        for n in names:
            o, v = agg(rows["oof"][n]), agg(rows["val"][n])
            do = o["lesion_sens"] - base["oof"]["lesion_sens"]
            dv = v["lesion_sens"] - base["val"]["lesion_sens"]
            ok = do >= BUDGET and dv >= BUDGET
            if ok:
                qual.append((o["fp_ml"], -o["dice"], n))
            print(f"  {n:28s}{do:+11.4f}{dv:+11.4f}{o['fp_ml']:11.2f}"
                  f"{o['dice']:10.4f}  {'yes' if ok else 'no'}")
        sel = sorted(qual)[0][2] if qual else None
        print(f"\n  RULE R selects: {sel}" if sel else
              "\n  RULE R selects nothing.")

        ent = {"thresholds": {p: list(t) for p, t in taus.items()},
               "qualifying": [q[2] for q in sorted(qual)], "selected": sel}
        if sel:
            for s in ("val", "test"):
                a, a0 = agg(rows[s][sel]), base[s]
                print(f"  {s:5s} M0  Dice {a0['dice']:.4f} FP {a0['fp_ml']:6.2f} "
                      f"sens {a0['lesion_sens']:.4f}   ->  {sel}: "
                      f"Dice {a['dice']:.4f} FP {a['fp_ml']:6.2f} "
                      f"sens {a['lesion_sens']:.4f}")
            bs = paired_boot(rows["test"][sel], rows["test"]["M0"],
                             ("lesion_sens", "fp_ml", "dice"))
            print("  test, selected - M0:")
            for k, v in bs.items():
                print(f"    {k:14s} {fmt(v)}")
            ent["test_bootstrap"] = bs

        # clinical view, on test, for the removal action at each point
        ent["lesion_tables"] = {}
        for p in POINTS:
            for n in (f"A remove [{p}]", f"A+B+reflectQ [{p}]"):
                t = lesion_table(books["test"], dec["test"][n])
                ent["lesion_tables"][n] = t
            print_lesion_table(f"test, A remove [{p}]",
                               ent["lesion_tables"][f"A remove [{p}]"], books["test"])

        # the budgeted operating point, per split
        print("\n  best FP reduction with lesion sensitivity within 1 point:")
        ent["budgeted"] = {}
        for s in ("oof", "val", "test"):
            b = budgeted_best(books[s], *sc[s], base[s])
            if b is None:
                print(f"    {s:5s} none")
                continue
            st, ta, tb = b
            print(f"    {s:5s} tau_A {ta:.2f} tau_B {tb:.2f} -> FP "
                  f"{base[s]['fp_ml']:6.2f} -> {st['fp_ml']:6.2f} mL "
                  f"({100*(st['fp_ml']/base[s]['fp_ml']-1):+.0f}%), "
                  f"Dice {st['dice']:.4f}, sens {st['lesion_sens']:.4f}")
            ent["budgeted"][s] = {"tau_a": ta, "tau_b": tb, **st}
        report[det] = ent

    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
