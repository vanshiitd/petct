#!/usr/bin/env python3
"""Build the ensemble probability map and three uncertainty maps per case.

Phase II, Part B. For each case, 16 softmax maps are produced from 2 models
(Dataset505 MAE-pretrained and its from-scratch twin, both fold 0,
checkpoint_final) times 8 flip combinations of the input, and reduced to four
maps.

Why the flips are done here rather than left to nnU-Net: nnU-Net's own
test-time mirroring averages *logits* inside each sliding-window patch and
softmaxes once at the end, so the 8 augmentations are already fused by the time
anything is observable and their disagreement -- which is exactly what U_stab
measures -- cannot be recovered. With `use_mirroring=False` each flip is a
separate prediction over the whole volume and the spread across them is
available.

One consequence to state rather than discover later: P_bar here is the mean of
softmax maps, while nnU-Net's own output is the softmax of mean logits, and the
flips are applied to the whole volume rather than per patch. So the ensemble
segmentation is close to, but not bit-identical with, the existing `*_prep`
predictions.

Cost stays reasonable because each case is preprocessed once and reused for all
16 passes, and because disabling nnU-Net's mirroring makes a single pass 8x
cheaper -- 16 passes cost about twice a default prediction, not 16 times.

Reductions, following the plan's eqs. 6-9, with M_m the mean of model m's 8
flips and P_bar = (M_1 + M_2) / 2:

    U_ent  = -(P log P + (1-P) log(1-P)),  P = P_bar            (eq. 6)
    U_epi  = variance between the two model means               (eqs. 7-9)
             = ((M_1 - M_2) / 2)^2, the population variance of two values
    U_stab = mean over models of the variance across that model's 8 flips

Only two members contribute to U_epi, so it is a difference rather than a
variance in any meaningful sense; it is kept in this form so more members can
be added later without changing the definition.

`confidence` = max(P_bar, 1 - P_bar) is not stored, being a pointwise function
of P_bar. U_ent is stored even though it is also derivable from P_bar, because
computing it from the float32 accumulator is slightly more accurate than
recomputing it from a float16 P_bar.

Maps are written in the *prepared* grid -- the one `labelsTs_prep` and the
existing `*_prep` predictions live in -- by pasting nnU-Net's internal
crop-to-nonzero back out, so everything downstream is already aligned.

    python scripts/make_uncertainty_maps.py --cases val_cases.txt \\
        --out D:\\data\\uncertainty\\val
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import torch

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
RESULTS = Path(r"D:\data\nnunet\results\Dataset505_AutoPET_MAEprep_v2")
MODELS = {
    "pretrained": RESULTS / "nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres",
    "scratch": RESULTS / "nnUNetTrainer_500epochs_scratch__nnUNetPlans_MAE__3d_fullres",
}
# all 8 subsets of the three spatial axes, identity first
FLIPS = [()] + [c for i in range(3) for c in itertools.combinations((0, 1, 2), i + 1)]


def build_predictor(model_dir: Path):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                        use_mirroring=False,          # the point: the flips are ours
                        device=torch.device("cuda"), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,),
                                           checkpoint_name="checkpoint_final.pth")
    return p


def case_files(case: str) -> list[str]:
    for sub in ("imagesTs", "imagesTr"):
        f0 = RAW / sub / f"{case}_0000.nii.gz"
        if f0.exists():
            return [str(f0), str(RAW / sub / f"{case}_0001.nii.gz")]
    raise FileNotFoundError(f"no prepared images for {case}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True,
                    help="file with one case id per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--models", default="pretrained,scratch",
                    help="comma-separated subset of MODELS. With a single model "
                         "there is no between-model term, so u_epi is not written "
                         "and u_stab is that model's own variance across flips.")
    args = ap.parse_args()

    os.environ.setdefault("nnUNet_raw", r"C:\nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", r"C:\nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", str(RESULTS.parent))
    os.environ.setdefault("nnUNet_extTrainer", r"D:\data\petct\nnunet_ext")

    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    cases = [c.strip() for c in args.cases.read_text().splitlines() if c.strip()]
    if args.limit:
        cases = cases[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    n_models = len([m for m in args.models.split(",") if m.strip()])
    print(f"{len(cases)} cases, {n_models} model(s) x {len(FLIPS)} flips "
          f"= {n_models * len(FLIPS)} passes each -> {args.out}", flush=True)
    print(flush=True)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; have {sorted(MODELS)}")
    predictors = {name: build_predictor(MODELS[name]) for name in wanted}
    first = next(iter(predictors.values()))
    pm, dj, cm = first.plans_manager, first.dataset_json, first.configuration_manager
    pre = DefaultPreprocessor(verbose=False)

    t_start = time.time()
    done = 0
    for idx, case in enumerate(cases, 1):
        out_file = args.out / f"{case}.npz"
        if out_file.exists() and not args.overwrite:
            print(f"  [{idx}/{len(cases)}] {case}: already", flush=True)
            continue
        t0 = time.time()
        data, _, props = pre.run_case(case_files(case), None, pm, cm, dj)

        shape = data.shape[1:]
        means, sq_means = {}, {}
        for name, predictor in predictors.items():
            s = np.zeros(shape, dtype=np.float32)
            q = np.zeros(shape, dtype=np.float32)
            for axes in FLIPS:
                arr = data if not axes else np.flip(data, [a + 1 for a in axes])
                logits = predictor.predict_logits_from_preprocessed_data(
                    torch.from_numpy(np.ascontiguousarray(arr)))
                prob = torch.softmax(logits.float(), 0)[1].cpu().numpy()
                del logits
                if axes:
                    prob = np.flip(prob, list(axes))
                s += prob
                q += prob * prob
                del prob
            means[name] = s / len(FLIPS)
            sq_means[name] = q / len(FLIPS)
            del s, q

        names = list(means)
        # E[p^2] - E[p]^2, clipped because floating point can make it a hair
        # negative where a model is perfectly consistent across flips
        vars_ = [np.clip(sq_means[n] - means[n] * means[n], 0.0, None) for n in names]
        u_stab = sum(vars_) / len(vars_)
        p_bar = sum(means[n] for n in names) / len(names)
        if len(names) == 1:
            # One member has no between-model variance to measure. Writing zeros
            # would look like confident agreement rather than an absent quantity,
            # so u_epi is omitted from the file entirely and readers must notice.
            u_epi = None
        else:
            m1, m2 = means[names[0]], means[names[1]]
            u_epi = ((m1 - m2) / 2.0) ** 2
        eps = 1e-7
        pc = np.clip(p_bar, eps, 1.0 - eps)
        u_ent = -(pc * np.log(pc) + (1.0 - pc) * np.log1p(-pc))
        del means, sq_means, vars_, pc

        # back out of nnU-Net's crop-to-nonzero, into the prepared grid
        full_shape = tuple(int(v) for v in props["shape_before_cropping"])
        bbox = props["bbox_used_for_cropping"]
        sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)

        def expand(a):
            out = np.zeros(full_shape, dtype=np.float16)
            out[sl] = a.astype(np.float16)
            return out

        payload = dict(p_bar=expand(p_bar), u_ent=expand(u_ent),
                       u_stab=expand(u_stab),
                       models=np.array(names),
                       shape=np.array(full_shape, dtype=np.int32),
                       bbox=np.array(bbox, dtype=np.int32),
                       spacing=np.array(props["spacing"], dtype=np.float64))
        if u_epi is not None:
            payload["u_epi"] = expand(u_epi)
        np.savez_compressed(out_file, **payload)
        done += 1
        rate = (time.time() - t_start) / done
        print(f"  [{idx}/{len(cases)}] {case}: {full_shape} "
              f"{time.time() - t0:.0f}s {out_file.stat().st_size / 1e6:.1f} MB "
              f"| ~{(len(cases) - idx) * rate / 3600:.1f} h left", flush=True)

    sizes = [f.stat().st_size for f in args.out.glob("*.npz")]
    if sizes:
        print(f"\n{len(sizes)} maps | {sum(sizes) / 1e9:.2f} GB total "
              f"| {np.mean(sizes) / 1e6:.1f} MB mean")
    else:
        print("\nnothing written")


if __name__ == "__main__":
    main()
