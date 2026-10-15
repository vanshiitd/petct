#!/usr/bin/env python3
"""Phase IV step 2: measure every strategy on every component it targets (OOF).

For each predicted component, each applicable strategy is applied to a *fresh*
copy of the state and the effect recorded: Delta-Dice for the case, lesions
lost and gained, Delta-FP volume, and Delta-Q from the independent quality
model. Rows are tagged with the failure classes the component is involved in,
so the output supports the per-failure x per-strategy table the rule-based
policy of eq. 30 is read from.

Two things make this affordable and exact at once.

*Local accounting.* A strategy only changes voxels inside one bounding box, so
Dice, false-positive volume and lesion hits are updated from that box alone
rather than rescored over the whole volume. Lesion hits need a little care: a
lesion is only lost if **no other** component still touches it, so the script
keeps a per-lesion touch count and updates it.

*Feature agreement.* Delta-Q is only meaningful if the features recomputed here
match the ones Q was trained on, which came from `build_candidates.py` via the
candidate CSV. The script therefore recomputes features for every *unmodified*
component and reports the agreement against the CSV before using them. A large
disagreement would make Delta-Q meaningless, so it is measured, not assumed.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext, REGISTRY, MIN_VOX
from q_model import QFEATS, WMEAN, read_cand

CPU_STRATEGIES = ["topology_remove", "topology_fill", "topology_merge",
                  "boundary", "suv41", "suv_percist",
                  "volume_grow", "volume_shrink"]


def q_vector(feats: list[dict], ctxinfo: dict) -> np.ndarray:
    """Assemble Q's input from per-component feature dicts (mirrors q_model)."""
    f = {k: np.nan for k in QFEATS}
    comps = [c for c in feats if c]
    tot = sum(c["n_vox"] for c in comps)
    f["q_vol_ml"] = sum(c["vol_ml"] for c in comps)
    f["q_ncomp"] = len(comps)
    if comps:
        sizes = np.array([c["n_vox"] for c in comps], float)
        vols = np.array([c["vol_ml"] for c in comps], float)
        f["q_logvol_max"] = float(np.log10(vols.max() + 1e-6))
        f["q_logvol_min"] = float(np.log10(vols.min() + 1e-6))
        sm = [c["suv_max"] for c in comps if np.isfinite(c["suv_max"])]
        f["q_suv_max"] = float(max(sm)) if sm else np.nan
        for col in WMEAN:
            v = np.array([c.get(col, np.nan) for c in comps], float)
            ok = np.isfinite(v)
            f[f"q_{col}"] = (float(np.sum(sizes[ok] * v[ok]) / sizes[ok].sum())
                             if ok.any() and sizes[ok].sum() else np.nan)
    else:
        f["q_logvol_max"] = f["q_logvol_min"] = -6.0
        f["q_suv_max"] = 0.0
        for col in WMEAN:
            f[f"q_{col}"] = 0.0
    f.update(ctxinfo)
    return np.array([f[k] for k in QFEATS], float)


def two_dices(iv: float, pv: float, gt: float) -> tuple[float, float]:
    """(Dice for the primary endpoint, Dice under Q's training convention).

    On a tumour-free case the ordinary Dice is 0/0 once a strategy empties the
    mask -- which is exactly the outcome removal exists to produce. Reporting
    that as nan poisons the mean and hides removal's best use, so two numbers
    are kept: the first is nan on tumour-free cases, because the pre-registered
    primary endpoint is Dice over tumour-POSITIVE cases only; the second
    follows q_model and scores an empty prediction on a tumour-free case as a
    perfect 1.0, which is the target Q was actually trained against and so the
    right basis for asking whether Q is gamed.
    """
    d_pos = (2.0 * iv / (pv + gt)) if gt > 0 and (pv + gt) > 0 else float("nan")
    d_q = (2.0 * iv / (pv + gt)) if (pv + gt) > 0 else 1.0
    return d_pos, d_q


def classes_of(diag: dict) -> tuple[dict, dict]:
    """component id -> set of failure classes, and lesion id -> classes."""
    cc = {}
    for c in diag["components"]:
        s = set()
        if c["spurious_class"]:
            s.add(c["spurious_class"])
        if c["merging"]:
            s.add("merging")
        cc[c["component_id"]] = s
    for l in diag["lesions"]:
        for k in ("under_seg", "over_seg", "boundary", "fragmentation"):
            if l[k]:
                for cid in l["comp_ids"]:
                    cc.setdefault(cid, set()).add(k)
    return cc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--diag", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--qpkl", type=Path, default=Path(r"D:\data\reports\q_model.pkl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="oof")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    diags = {d["case"]: d for d in pickle.load(open(args.diag, "rb"))}
    cand, ctxmap = read_cand(args.cand)
    with open(args.qpkl, "rb") as fh:
        Q = pickle.load(fh)
    from q_model import patient_of
    qfull = Q["full"]
    qfolds = Q["folds"]

    def qpred(case, vec):
        if args.split == "oof":
            p = patient_of(case)
            for grp, m in qfolds:
                if p in grp:
                    return float(m.predict(vec[None])[0])
        return float(qfull.predict(vec[None])[0])

    cases = sorted(diags)
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | strategies {CPU_STRATEGIES} + recover",
          flush=True)

    rows = []
    agree = {k: [] for k in ("p_mean", "suv_mean", "sphericity", "dist_physio_mm")}
    for n_done, case in enumerate(cases, 1):
        d = diags[case]
        ctx = CaseContext(case, args.maps)
        cls = classes_of(d)

        base_feats = {}
        for cid in ctx.comp_ids:
            sl = ctx.comp_slice(cid)
            base_feats[cid] = ctx.features(sl, ctx.labels[sl] == cid)
        # agreement with what Q was trained on
        for cid, f in base_feats.items():
            row = cand.get((case, "pred", cid))
            if row and f:
                for k in agree:
                    a, b = f.get(k), row.get(k)
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        agree[k].append(abs(a - b))

        cinfo = {"c_npred": ctxmap.get(case, {}).get("c_npred", 0),
                 "c_nrec": ctxmap.get(case, {}).get("c_nrec", 0),
                 "c_liver_suv": ctx.liver_suv,
                 "c_vol_m0_ml": sum(f["vol_ml"] for f in base_feats.values() if f)}
        base_q = qpred(case, q_vector(list(base_feats.values()), cinfo))
        base = ctx.score()
        base_pv = base["pred_ml"] / ctx.vml
        base_iv = (base["dice"] * (base_pv + ctx.gt_vox) / 2
                   if base["dice"] == base["dice"] else 0.0)
        b_pos, b_q = two_dices(base_iv, base_pv, ctx.gt_vox)

        # per-lesion touch counts, so a lesion is only "lost" when nothing holds it
        touch = {}
        for c in d["components"]:
            for g in c["gt_ids"]:
                touch[g] = touch.get(g, 0) + 1
        small_hit = set(np.unique(ctx.g_lab[ctx.small]).tolist()) - {0} \
            if ctx.small.any() else set()
        for g in small_hit:
            touch[g] = touch.get(g, 0) + 1

        gt_sl = ctx.gt
        for cid in list(ctx.comp_ids):
            sl0 = ctx.comp_slice(cid)
            old_local = (ctx.labels[sl0] == cid)
            old_n = int(old_local.sum())
            old_i = int((old_local & gt_sl[sl0]).sum())
            old_g = set(np.unique(ctx.g_lab[sl0][old_local]).tolist()) - {0}
            for name in CPU_STRATEGIES:
                sl, new_local = REGISTRY[name](ctx, cid)
                if sl is None:
                    continue
                # re-express the old component on the (possibly padded) box
                old_on_sl = (ctx.labels[sl] == cid)
                o_n = int(old_on_sl.sum())
                o_i = int((old_on_sl & gt_sl[sl]).sum())
                o_g = set(np.unique(ctx.g_lab[sl][old_on_sl]).tolist()) - {0}
                n_n = int(new_local.sum())
                n_i = int((new_local & gt_sl[sl]).sum())
                n_g = set(np.unique(ctx.g_lab[sl][new_local]).tolist()) - {0}
                if n_n == o_n and n_i == o_i and n_g == o_g:
                    continue                     # the strategy did nothing here
                dp, di = n_n - o_n, n_i - o_i
                pv, iv = base_pv + dp, base_iv + di
                d_pos, d_q = two_dices(iv, pv, ctx.gt_vox)
                lost = sum(1 for g in o_g - n_g if touch.get(g, 0) <= 1)
                gained = sum(1 for g in n_g - o_g if touch.get(g, 0) == 0)
                was_fp, is_fp = o_i == 0, n_i == 0
                dfp = ((n_n if is_fp else 0) - (o_n if was_fp else 0)) * ctx.vml
                nf = ctx.features(sl, new_local)
                feats = dict(base_feats)
                feats[cid] = nf
                q = qpred(case, q_vector(list(feats.values()), cinfo))
                rows.append({
                    "case": case, "component_id": cid, "strategy": name,
                    "classes": "|".join(sorted(cls.get(cid, set()))) or "none",
                    "vol_ml_before": o_n * ctx.vml, "vol_ml_after": n_n * ctx.vml,
                    "d_dice": d_pos - b_pos, "d_qdice": d_q - b_q,
                    "lesions_lost": lost,
                    "lesions_gained": gained, "d_fp_ml": dfp,
                    "d_q": q - base_q, "base_dice": base["dice"],
                    "gt_positive": bool(ctx.gt_vox > 0)})

        # recovery acts on the sub-threshold pool, not on existing components
        for rid, (sl, local) in ctx.recovery_candidates().items():
            n_n = int(local.sum())
            n_i = int((local & gt_sl[sl]).sum())
            n_g = set(np.unique(ctx.g_lab[sl][local]).tolist()) - {0}
            pv, iv = base_pv + n_n, base_iv + n_i
            d_pos, d_q = two_dices(iv, pv, ctx.gt_vox)
            gained = sum(1 for g in n_g if touch.get(g, 0) == 0)
            nf = ctx.features(sl, local)
            feats = list(base_feats.values()) + [nf]
            q = qpred(case, q_vector(feats, cinfo))
            rows.append({
                "case": case, "component_id": -rid, "strategy": "recover",
                "classes": "missed" if gained else "none",
                "vol_ml_before": 0.0, "vol_ml_after": n_n * ctx.vml,
                "d_dice": d_pos - b_pos, "d_qdice": d_q - b_q, "lesions_lost": 0,
                "lesions_gained": gained, "d_fp_ml": (n_n * ctx.vml) if n_i == 0 else 0.0,
                "d_q": q - base_q, "base_dice": base["dice"],
                "gt_positive": bool(ctx.gt_vox > 0)})

        if n_done % 25 == 0 or n_done == len(cases):
            print(f"  {n_done}/{len(cases)}  rows {len(rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")

    print("\nfeature agreement with the candidate CSV Q was trained on "
          "(mean |difference| over unmodified components):")
    for k, v in agree.items():
        print(f"  {k:18s} {np.mean(v):.6g}   n={len(v)}")


if __name__ == "__main__":
    main()
