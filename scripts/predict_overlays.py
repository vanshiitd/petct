#!/usr/bin/env python3
"""Render predicted vs ground-truth tumour masks for a few test patients.

Rebuilds the same seeded train/test split used in fine-tuning, runs
sliding-window inference with a trained checkpoint on lesion-positive test
patients, and saves overlay figures for slides or reports.

To avoid cherry-picking, it scores --scan lesion-positive test patients (in
split order) and plots the best, median and worst of them by per-patient Dice.
Every scored patient is listed in dice_per_patient.csv.

Each figure shows, left to right:
  1. PET coronal maximum-intensity projection with the ground truth (green)
  2. the same projection with the prediction (red)
  3. an axial CT slice through the largest ground-truth area, with both outlines

Examples:
    python scripts/predict_overlays.py --arch small \\
        --checkpoint runs/best_seg_small_100pct_mae.pth --data-root D:/data/autopet_nifti

    python scripts/predict_overlays.py --arch nnunet \\
        --checkpoint runs/best_seg_nnunet_100pct_mae.pth --data-root D:/data/autopet_nifti \\
        --patients PETCT_0011f3deaf PETCT_07b7e9abfc
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402

from petct import config, devices  # noqa: E402
from petct.config import ARCHS, SPLITS, get_arch  # noqa: E402
from petct.models import build_backbone  # noqa: E402
from petct.splits import build_subject_index, split_subjects  # noqa: E402
from petct.transforms import AutoPETPreprocessd  # noqa: E402


def load_model(arch, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load either a best_seg_* file (bare state_dict) or a latest_seg_* file."""
    model = build_backbone(arch, out_channels=config.NUM_CLASSES)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)  # strict: a mismatched arch must fail loudly
    return model.to(device).eval()


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom else float("nan")


def head_up(pet_path: str) -> bool:
    """True if array index 0 is the inferior end (so origin='lower' puts the head on top)."""
    return sitk.ReadImage(pet_path).GetDirection()[8] > 0


def render(case: dict, out_path: Path) -> None:
    pet, ct, gt, pred = case["pet"], case["ct"], case["gt"], case["pred"]
    origin = "lower" if case["head_up"] else "upper"

    mip = pet.max(axis=1)  # (Z, X): project over the anterior-posterior axis
    gt_mip, pred_mip = gt.max(axis=1), pred.max(axis=1)
    lo, hi = np.percentile(mip, [1, 99.5])

    z = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
    ct_lo, ct_hi = np.percentile(ct[z], [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, mask, color, label in ((axes[0], gt_mip, "#1a9850", "Ground truth"),
                                   (axes[1], pred_mip, "#d73027", "Prediction")):
        ax.imshow(mip, cmap="gray_r", vmin=lo, vmax=hi, origin=origin, aspect="auto")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=matplotlib.colors.ListedColormap([color]),
                  alpha=0.75, origin=origin, aspect="auto")
        ax.set_title(f"PET MIP + {label}", fontsize=12)
        ax.axis("off")

    axes[2].imshow(ct[z], cmap="gray", vmin=ct_lo, vmax=ct_hi)
    if gt[z].any():
        axes[2].contour(gt[z], levels=[0.5], colors="#1a9850", linewidths=1.6)
    if pred[z].any():
        axes[2].contour(pred[z], levels=[0.5], colors="#d73027", linewidths=1.6)
    axes[2].set_title("Axial CT: green = truth, red = prediction", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(f"{case['patient']}   Dice {case['dice']:.3f}   "
                 f"(truth {int(gt.sum())} vox, predicted {int(pred.sum())} vox)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ARCHS))
    p.add_argument("--checkpoint", type=Path, required=True, help="best_seg_*.pth or latest_seg_*.pth")
    p.add_argument("--data-root", type=Path, default=config.PATHS.autopet_root)
    p.add_argument("--split", choices=sorted(SPLITS), default="full")
    p.add_argument("--patients", nargs="*", default=None,
                   help="explicit patient IDs to render (skips best/median/worst selection)")
    p.add_argument("--scan", type=int, default=12,
                   help="how many lesion-positive test patients to score before picking best/median/worst")
    p.add_argument("--out", type=Path, default=Path("overlays"))
    p.add_argument("--device", default="auto")
    p.add_argument("--sw-batch-size", type=int, default=1)
    args = p.parse_args()

    arch = get_arch(args.arch)
    device = devices.resolve_device(args.device)
    print(f"Device: {devices.describe(device)}")
    model = load_model(arch, args.checkpoint, device)

    subjects = build_subject_index(args.data_root)
    _, test_files = split_subjects(subjects, 1.0, config.get_split(args.split))
    by_patient = {Path(f["pet_path"]).relative_to(args.data_root).parts[0]: f for f in test_files}

    if args.patients:
        missing = [pid for pid in args.patients if pid not in by_patient]
        if missing:
            print(f"Note: not in the test split, rendering anyway: {', '.join(missing)}")
        for pid in missing:
            d = args.data_root / pid
            by_patient[pid] = {"pet_path": str(d / "PET.nii.gz"), "ct_path": str(d / "CT_resample.nii.gz"),
                               "seg_path": str(d / "tumorSeg.nii.gz")}
        queue = list(args.patients)
    else:
        queue = list(by_patient)

    prep = AutoPETPreprocessd(keys=["image", "label"])
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for pid in queue:
        if not args.patients and len(cases) >= args.scan:
            break
        d = prep(dict(by_patient[pid]))
        gt = d["label"][0] > 0
        if not args.patients and not gt.any():
            continue  # lesion-free: Dice is undefined, nothing to show
        image = torch.from_numpy(d["image"]).unsqueeze(0).to(device)
        with torch.no_grad(), devices.amp_autocast(device):
            logits = sliding_window_inference(image, config.ROI_SIZE, args.sw_batch_size, model,
                                              overlap=config.SLIDING_WINDOW_OVERLAP)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(bool)
        del image, logits
        devices.empty_cache(device)

        case = {"patient": pid, "pet": d["image"][0], "ct": d["image"][1], "gt": gt, "pred": pred,
                "dice": dice(pred, gt), "head_up": head_up(by_patient[pid]["pet_path"])}
        cases.append(case)
        print(f"  {pid}: Dice {case['dice']:.3f}  truth {int(gt.sum())} vox  pred {int(pred.sum())} vox")

    if not cases:
        raise SystemExit("No lesion-positive test patients found to render.")

    with open(args.out / "dice_per_patient.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "dice", "truth_voxels", "pred_voxels"])
        for c in cases:
            w.writerow([c["patient"], f"{c['dice']:.4f}", int(c["gt"].sum()), int(c["pred"].sum())])

    if args.patients:
        chosen = [(c["patient"], c) for c in cases]
    else:
        ranked = sorted(cases, key=lambda c: np.nan_to_num(c["dice"], nan=-1.0))
        picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
        chosen, seen = [], set()
        for tag, c in picks.items():
            if c["patient"] not in seen:  # with few cases, best/median can coincide
                seen.add(c["patient"])
                chosen.append((f"{tag}_{c['patient']}", c))
        scores = np.array([c["dice"] for c in cases], dtype=float)
        print(f"\nScored {len(cases)} lesion-positive test patients: "
              f"mean Dice {np.nanmean(scores):.3f}, median {np.nanmedian(scores):.3f}")

    for name, c in chosen:
        path = args.out / f"{args.arch}_{name}.png"
        render(c, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
