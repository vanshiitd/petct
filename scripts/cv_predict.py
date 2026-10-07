#!/usr/bin/env python3
"""Out-of-fold and ensemble predictions from the 5 cross-validation folds.

Two modes, both writing the same four maps per case as
`make_uncertainty_maps.py` so everything downstream is unchanged:

  --mode oof       For every training scan, predict it with the one fold model
                   that never trained on it. Reads `splits_final.json` to find
                   which fold held each scan out, so the assignment is taken
                   from the file the training actually used rather than
                   recomputed and assumed to agree.

  --mode ensemble  Predict val or test with all 5 fold models at once. P_bar is
                   the mean over 5 models x 8 flips, U_epi is the variance
                   between the 5 model means -- which, unlike the two-member
                   version in Phase II, is a variance with enough members to
                   mean something -- and U_stab is the mean within-model spread
                   across flips.

Flips are fused by averaging logits, matching nnU-Net's own mirroring, because
averaging probabilities instead predicts about 10% more volume and would make
every comparison against an nnU-Net baseline partly an artefact of the fusion
rule. The per-flip softmax is still used for U_stab.

    python scripts/cv_predict.py --mode oof --out D:\\data\\cv\\oof
    python scripts/cv_predict.py --mode ensemble --cases reports/cases_test.txt \\
        --out D:\\data\\cv\\test
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
RESULTS = Path(r"D:\data\nnunet\results\Dataset505_AutoPET_MAEprep_v2")
TRAINER = "nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres"
SPLITS = Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json")
CV_FOLDS = (3, 4, 5, 6, 7)
FLIPS = [()] + [c for i in range(3) for c in itertools.combinations((0, 1, 2), i + 1)]


def build_predictor(fold: int):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        device=torch.device("cuda"), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(RESULTS / TRAINER), use_folds=(fold,),
                                           checkpoint_name="checkpoint_final.pth")
    return p


def case_files(case: str) -> list[str]:
    for sub in ("imagesTs", "imagesTr"):
        f0 = RAW / sub / f"{case}_0000.nii.gz"
        if f0.exists():
            return [str(f0), str(RAW / sub / f"{case}_0001.nii.gz")]
    raise FileNotFoundError(case)


def run_model(predictor, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (P from mean logits, sum of per-flip softmax, sum of squares)."""
    shape = data.shape[1:]
    s = np.zeros(shape, dtype=np.float32)
    q = np.zeros(shape, dtype=np.float32)
    lsum = np.zeros((2,) + shape, dtype=np.float32)
    for axes in FLIPS:
        arr = data if not axes else np.flip(data, [a + 1 for a in axes])
        lg = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(np.ascontiguousarray(arr))).float()
        if axes:
            lg = torch.flip(lg, [a + 1 for a in axes])
        pr = torch.softmax(lg, 0)[1].cpu().numpy()
        lsum += lg.cpu().numpy()
        del lg
        s += pr
        q += pr * pr
        del pr
    lm = lsum / len(FLIPS)
    lm -= lm.max(axis=0, keepdims=True)
    e = np.exp(lm)
    return (e[1] / e.sum(axis=0)).astype(np.float32), s / len(FLIPS), q / len(FLIPS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("oof", "ensemble"), required=True)
    ap.add_argument("--cases", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("nnUNet_raw", r"C:\nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", r"C:\nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", str(RESULTS.parent))
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    folds = json.loads(SPLITS.read_text())
    if args.mode == "oof":
        owner = {}
        for k in CV_FOLDS:
            for s in folds[k]["val"]:
                if s in owner:
                    raise SystemExit(f"{s} held out by folds {owner[s]} and {k}")
                owner[s] = k
        cases = sorted(owner)
        print(f"out-of-fold: {len(cases)} training scans across folds {CV_FOLDS}")
    else:
        cases = [c.strip() for c in args.cases.read_text().splitlines() if c.strip()]
        owner = None
        print(f"ensemble of {len(CV_FOLDS)} folds over {len(cases)} cases")
    if args.limit:
        cases = cases[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    pre = DefaultPreprocessor(verbose=False)
    predictors: dict[int, object] = {}
    first = None
    t0, done = time.time(), 0
    for i, case in enumerate(cases, 1):
        out_file = args.out / f"{case}.npz"
        if out_file.exists():
            print(f"  [{i}/{len(cases)}] {case}: already", flush=True)
            continue
        use = [owner[case]] if owner else list(CV_FOLDS)
        for k in use:
            if k not in predictors:
                predictors[k] = build_predictor(k)
                first = first or predictors[k]
        data, _, props = pre.run_case(case_files(case), None, first.plans_manager,
                                      first.configuration_manager, first.dataset_json)
        P, S, Q = [], [], []
        for k in use:
            p_, s_, q_ = run_model(predictors[k], data)
            P.append(p_); S.append(s_); Q.append(q_)
        p_bar = np.mean(P, axis=0)
        u_stab = np.mean([np.clip(q - s * s, 0.0, None) for s, q in zip(S, Q)], axis=0)
        u_epi = (np.var(P, axis=0).astype(np.float32) if len(P) > 1 else None)
        eps = 1e-7
        pc = np.clip(p_bar, eps, 1 - eps)
        u_ent = -(pc * np.log(pc) + (1 - pc) * np.log1p(-pc)).astype(np.float32)

        full = tuple(int(v) for v in props["shape_before_cropping"])
        bbox = props["bbox_used_for_cropping"]
        sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)

        def expand(a):
            o = np.zeros(full, dtype=np.float16)
            o[sl] = a.astype(np.float16)
            return o

        payload = dict(p_bar=expand(p_bar), u_ent=expand(u_ent), u_stab=expand(u_stab),
                       folds=np.array(use), combine=np.array("logit"),
                       shape=np.array(full, dtype=np.int32),
                       bbox=np.array(bbox, dtype=np.int32),
                       spacing=np.array(props["spacing"], dtype=np.float64))
        if u_epi is not None:
            payload["u_epi"] = expand(u_epi)
        np.savez_compressed(out_file, **payload)
        done += 1
        rate = (time.time() - t0) / done
        print(f"  [{i}/{len(cases)}] {case}: folds {use} {full} "
              f"{out_file.stat().st_size / 1e6:.1f} MB "
              f"| ~{(len(cases) - i) * rate / 3600:.1f} h left", flush=True)

    sizes = [f.stat().st_size for f in args.out.glob("*.npz")]
    print(f"\n{len(sizes)} maps | {sum(sizes) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
