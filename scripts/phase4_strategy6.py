#!/usr/bin/env python3
"""Phase IV strategy 6: re-predict a crop around an uncertain component (GPU).

A component flagged as uncertain is re-segmented by running the network again
on a crop around it with heavier test-time augmentation than the whole-scan
pass used -- 8 flips times a set of small shifts -- and averaging logits, not
softmax (Phase II: averaging softmax predicts about 10% more volume). The
component is then replaced by the re-predicted mask inside its own box; nothing
outside the box moves.

**Which models may vote is not a free choice.** For an out-of-fold case, four
of the five fold models trained on that patient, so an ensemble of all five
would leak. Each out-of-fold map records the single held-out fold in its
`folds` field, and only that model is used here. On val and test, where no fold
model saw the case, all five vote. The script reads the fold list from the map
rather than taking it on trust.

This is the expensive strategy, so it is capped: `--max-components` targets
only the components the diagnosis model considers most uncertain, and the cap
is reported with the results so the per-strategy table says what it is based on.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from phase4_strategy_table import q_vector, two_dices
from q_model import read_cand, patient_of
from cv_predict import build_predictor, case_files, FLIPS

SHIFTS = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (-2, 0, 0), (0, -2, 0)]
PAD_MM = 24.0


def crop_box(sl, shape, spacing, patch=None):
    """Component box widened by PAD_MM, then out to at least one patch.

    PAD_MM alone is not enough. The network's patch is 192x96x128 and
    `predict_logits_from_preprocessed_data` zero-pads anything smaller up to
    it, so a 24 mm box around a 0.2 mL component arrives as a bright speck in
    a uniform block with no anatomy around it: measured P.max() 3e-8 against
    0.997 for the same voxels in the whole-scan pass. The crop therefore has
    to carry at least a patch of real context, centred on the component.
    """
    pad = tuple(int(np.ceil(PAD_MM / s)) for s in spacing)
    out = []
    for i, (sli, n, pv) in enumerate(zip(sl, shape, pad)):
        a, b = max(0, sli.start - pv), min(n, sli.stop + pv)
        if patch is not None:
            want = min(int(patch[i]), n)
            if b - a < want:
                c = (a + b) // 2
                a = max(0, c - want // 2)
                b = min(n, a + want)
                a = max(0, b - want)
        out.append(slice(a, b))
    return tuple(out)


def predict_crop(predictors, data_crop) -> np.ndarray:
    """Mean logits over models x flips x shifts, softmaxed once."""
    acc = None
    n = 0
    for pred in predictors:
        for sh in SHIFTS:
            shifted = np.roll(data_crop, sh, axis=(1, 2, 3)) if any(sh) else data_crop
            for axes in FLIPS:
                arr = shifted if not axes else np.flip(shifted, [a + 1 for a in axes])
                lg = pred.predict_logits_from_preprocessed_data(
                    torch.from_numpy(np.ascontiguousarray(arr))).float()
                if axes:
                    lg = torch.flip(lg, [a + 1 for a in axes])
                lg = lg.cpu().numpy()
                if any(sh):
                    lg = np.roll(lg, [-v for v in sh], axis=(1, 2, 3))
                acc = lg if acc is None else acc + lg
                n += 1
    lm = acc / n
    lm -= lm.max(axis=0, keepdims=True)
    e = np.exp(lm)
    return (e[1] / e.sum(axis=0)).astype(np.float32)


class CaseRepredictor:
    """`roi_repredict` as a REGISTRY-compatible strategy, bound to one case.

    Phase IV measured this strategy only as a per-component counterfactual,
    because the expensive parts -- preprocessing the case and building the fold
    predictors -- are per case, not per component, and the strategy table could
    amortise them while the agent loop could not. Holding them on an instance
    lets the agent call `self(ctx, cid)` like any other strategy, so Phase V can
    measure what Phase IV could not: whether the per-component gains survive
    composing several of them on one case.

    Construct once per case, then call as a strategy. Returns (None, None) when
    the component lies outside the preprocessed crop box, which the agent reads
    as "not applicable" exactly as it does for the CPU strategies.
    """

    def __init__(self, preds, data, off, ext):
        self.preds = preds
        self.data = data
        self.off = off
        self.ext = ext

    def __call__(self, ctx, cid: int, **kw):
        if cid not in ctx.comp_ids:
            return None, None
        sl = crop_box(ctx.comp_slice(cid), ctx.shape, ctx.spacing,
                      self.preds[0].configuration_manager.patch_size)
        sl_pre = tuple(slice(max(0, a.start - o), min(e, a.stop - o))
                       for a, o, e in zip(sl, self.off, self.ext))
        if any(x.stop <= x.start for x in sl_pre):
            return None, None
        sl = tuple(slice(x.start + o, x.stop + o)
                   for x, o in zip(sl_pre, self.off))
        P = predict_crop(self.preds, self.data[(slice(None),) + sl_pre])
        old_local = ctx.labels[sl] == cid
        new_local = P > 0.5
        lab, nl = ndi.label(new_local)
        if nl and old_local.any():
            keep = np.unique(lab[old_local])
            keep = keep[keep > 0]
            new_local = np.isin(lab, keep) if keep.size else old_local
        other = (ctx.labels[sl] > 0) & ~old_local
        new_local &= ~other
        return sl, new_local


def build_repredictor(case: str, maps: Path, split: str, pre, get_predictors):
    """Preprocess one case and bind a CaseRepredictor to it."""
    z = np.load(maps / f"{case}.npz")
    folds = [int(v) for v in z["folds"]]
    if split == "oof" and len(folds) != 1:
        raise SystemExit(f"{case}: out-of-fold map lists {folds}; expected one "
                         f"held-out fold. Refusing to leak.")
    preds = get_predictors(folds)
    data, _, props = pre.run_case(case_files(case), None,
                                  preds[0].plans_manager,
                                  preds[0].configuration_manager,
                                  preds[0].dataset_json)
    bbox = props["bbox_used_for_cropping"]
    off = [int(b[0]) for b in bbox]
    ext = tuple(int(b[1]) - int(b[0]) for b in bbox)
    if tuple(data.shape[1:]) != ext:
        raise SystemExit(f"{case}: preprocessed {data.shape[1:]} != crop box "
                         f"{ext}; the plan resamples, so component boxes cannot "
                         f"be carried over unchanged.")
    return CaseRepredictor(preds, data, off, ext)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--diag", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--qpkl", type=Path, default=Path(r"D:\data\reports\q_model.pkl"))
    ap.add_argument("--split", default="oof")
    ap.add_argument("--max-components", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from phase4_diag_model import PRED_CLASSES
    from phase4_agent import observable_rows, observable_index
    D = pickle.load(open(args.diagpkl, "rb"))["pred"]
    obs = observable_index(args.cand, "pred")
    Q = pickle.load(open(args.qpkl, "rb"))
    cand, ctxmap = read_cand(args.cand)
    diags = {d["case"]: d for d in pickle.load(open(args.diag, "rb"))}

    # rank every component by predicted uncertainty-ish failure, then cap
    print("ranking components by predicted failure (boundary/under/over) ...",
          flush=True)
    ranked = []
    for case in sorted(diags):
        ids, X = observable_rows(obs, case, D["feats"], D["median"])
        if not ids:
            continue
        sc = np.zeros(len(ids))
        for k in ("boundary", "under_seg", "over_seg"):
            m = D["models"].get(k)
            if m is not None:
                sc = np.maximum(sc, m.predict_proba(X)[:, 1])
        for cid, s in zip(ids, sc):
            ranked.append((float(s), case, cid))
    ranked.sort(reverse=True)
    sel = ranked[: args.max_components]
    by_case = {}
    for s, case, cid in sel:
        by_case.setdefault(case, []).append(cid)
    print(f"{len(ranked)} components, re-predicting the top {len(sel)} "
          f"over {len(by_case)} cases (cap --max-components {args.max_components})",
          flush=True)

    import os
    os.environ.setdefault("nnUNet_raw", r"C:\nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", r"C:\nnunet_preprocessed")
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    pre = DefaultPreprocessor(verbose=False)

    cache = {}

    def get_predictors(folds):
        key = tuple(sorted(folds))
        if key not in cache:
            cache[key] = [build_predictor(f) for f in key]
        return cache[key]

    rows = []
    t0 = time.time()
    for n, (case, cids) in enumerate(sorted(by_case.items()), 1):
        z = np.load(args.maps / f"{case}.npz")
        folds = [int(v) for v in z["folds"]]
        if args.split == "oof" and len(folds) != 1:
            raise SystemExit(f"{case}: out-of-fold map lists {folds}; expected one "
                             f"held-out fold. Refusing to leak.")
        preds = get_predictors(folds)
        ctx = CaseContext(case, args.maps)
        # The network takes nnU-Net *preprocessed* data, not raw intensities:
        # fed the raw SUV/HU arrays it returns an empty mask for every crop.
        # run_case crops to the nonzero box, so preprocessed coordinates are
        # original coordinates minus that box's origin. cv_predict writes the
        # maps back through the same box, which is why the two line up -- and
        # the extent is checked rather than assumed, since a plan whose target
        # spacing differed from the data would silently shift every crop.
        data, _, props = pre.run_case(case_files(case), None,
                                      preds[0].plans_manager,
                                      preds[0].configuration_manager,
                                      preds[0].dataset_json)
        bbox = props["bbox_used_for_cropping"]
        off = [int(b[0]) for b in bbox]
        ext = tuple(int(b[1]) - int(b[0]) for b in bbox)
        if tuple(data.shape[1:]) != ext:
            raise SystemExit(f"{case}: preprocessed {data.shape[1:]} != crop box "
                             f"{ext}; the plan resamples, so component boxes "
                             f"cannot be carried over unchanged.")
        base_feats = {cid: ctx.features(ctx.comp_slice(cid),
                                        ctx.labels[ctx.comp_slice(cid)] == cid)
                      for cid in ctx.comp_ids}
        cinfo = {"c_npred": ctxmap.get(case, {}).get("c_npred", 0),
                 "c_nrec": ctxmap.get(case, {}).get("c_nrec", 0),
                 "c_liver_suv": ctx.liver_suv,
                 "c_vol_m0_ml": sum(f["vol_ml"] for f in base_feats.values() if f)}

        def qof(feats):
            vec = q_vector(list(feats.values()), cinfo)
            if args.split == "oof":
                p = patient_of(case)
                for grp, m in Q["folds"]:
                    if p in grp:
                        return float(m.predict(vec[None])[0])
            return float(Q["full"].predict(vec[None])[0])

        base_q = qof(base_feats)
        base = ctx.score()
        # A lesion counts as lost only when NOTHING else still touches it --
        # the same per-lesion touch count phase4_strategy_table keeps. Without
        # it a lesion held by three components is scored as lost three times
        # and the policy's safety guard drops the strategy on a miscount.
        touch = {}
        for c in diags[case]["components"]:
            for g in c["gt_ids"]:
                touch[g] = touch.get(g, 0) + 1
        if ctx.small.any():
            for g in set(np.unique(ctx.g_lab[ctx.small]).tolist()) - {0}:
                touch[g] = touch.get(g, 0) + 1
        for cid in cids:
            if cid not in ctx.comp_ids:
                continue
            sl = crop_box(ctx.comp_slice(cid), ctx.shape, ctx.spacing,
                          preds[0].configuration_manager.patch_size)
            sl_pre = tuple(slice(max(0, a.start - o), min(e, a.stop - o))
                           for a, o, e in zip(sl, off, ext))
            if any(x.stop <= x.start for x in sl_pre):
                continue                 # component lies outside the crop box
            # keep the two views the same size: the original-space box is
            # trimmed to whatever survived the crop box
            sl = tuple(slice(x.start + o, x.stop + o)
                       for x, o in zip(sl_pre, off))
            sub = data[(slice(None),) + sl_pre]
            P = predict_crop(preds, sub)
            old_local = ctx.labels[sl] == cid
            new_local = P > 0.5
            lab, nl = ndi.label(new_local)
            if nl and old_local.any():
                keep = np.unique(lab[old_local])
                keep = keep[keep > 0]
                new_local = np.isin(lab, keep) if keep.size else old_local
            other = (ctx.labels[sl] > 0) & ~old_local
            new_local &= ~other
            o_n, n_n = int(old_local.sum()), int(new_local.sum())
            o_i = int((old_local & ctx.gt[sl]).sum())
            n_i = int((new_local & ctx.gt[sl]).sum())
            o_g = set(np.unique(ctx.g_lab[sl][old_local]).tolist()) - {0}
            n_g = set(np.unique(ctx.g_lab[sl][new_local]).tolist()) - {0}
            base_pv = base["pred_ml"] / ctx.vml
            base_iv = (base["dice"] * (base_pv + ctx.gt_vox) / 2
                       if base["dice"] == base["dice"] else 0.0)
            b_pos, b_q = two_dices(base_iv, base_pv, ctx.gt_vox)
            pv = base_pv + (n_n - o_n)
            iv = base_iv + (n_i - o_i)
            d_pos, d_q_dice = two_dices(iv, pv, ctx.gt_vox)
            nf = ctx.features(sl, new_local) if n_n else None
            f2 = dict(base_feats)
            if nf is None:
                f2.pop(cid, None)
            else:
                f2[cid] = nf
            d = diags[case]
            cls = set()
            for l in d["lesions"]:
                if cid in l["comp_ids"]:
                    for k in ("under_seg", "over_seg", "boundary", "fragmentation"):
                        if l[k]:
                            cls.add(k)
            for c in d["components"]:
                if c["component_id"] == cid:
                    if c["spurious_class"]:
                        cls.add(c["spurious_class"])
                    if c["merging"]:
                        cls.add("merging")
            rows.append({
                "case": case, "component_id": cid, "strategy": "roi_repredict",
                "classes": "|".join(sorted(cls)) or "none",
                "vol_ml_before": o_n * ctx.vml, "vol_ml_after": n_n * ctx.vml,
                "d_dice": d_pos - b_pos, "d_qdice": d_q_dice - b_q,
                "lesions_lost": sum(1 for g in o_g - n_g if touch.get(g, 0) <= 1),
                "lesions_gained": sum(1 for g in n_g - o_g if touch.get(g, 0) == 0),
                "d_fp_ml": ((n_n if n_i == 0 else 0) - (o_n if o_i == 0 else 0)) * ctx.vml,
                "d_q": qof(f2) - base_q, "base_dice": base["dice"],
                "gt_positive": bool(ctx.gt_vox > 0)})
        if n % 10 == 0 or n == len(by_case):
            el = time.time() - t0
            print(f"  {n}/{len(by_case)} cases, {len(rows)} components, "
                  f"{el/max(len(rows),1):.1f} s/component", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    d = np.array([r["d_dice"] for r in rows if r["gt_positive"]], float)
    print(f"mean dDice (tumour-positive) {np.nanmean(d):+.4f}, improved {int((d>0).sum())}, "
          f"worsened {int((d<0).sum())}, "
          f"lesions lost {sum(r['lesions_lost'] for r in rows)}, "
          f"gained {sum(r['lesions_gained'] for r in rows)}")


if __name__ == "__main__":
    main()
