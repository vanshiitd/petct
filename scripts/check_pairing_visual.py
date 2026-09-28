#!/usr/bin/env python3
"""Eyeball whether a scan's PET, CT and mask actually belong together.

A mask paired with the wrong visit still produces a perfectly well-formed NIfTI,
and a CT from the wrong visit can resample to a uniform -1000 HU block. Neither
shows up in the file counts, so this renders what the numbers cannot:

  * a coronal PET maximum-intensity projection with the mask outlined over it --
    the mask should sit on the PET's hot spots, not float beside them
  * a coronal CT MIP and one axial CT slice -- should show real anatomy, not the
    flat grey of an all-air volume

    python scripts/check_pairing_visual.py --data-root D:\\data\\autopet_nifti \\
        --out D:\\data\\reports\\pairing_check PETCT_234f8427c0 PETCT_86153b2974
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def load(scan_dir: Path):
    def arr(name):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(scan_dir / name)))
    return arr("PET.nii.gz"), arr("CT_resample.nii.gz"), arr("tumorSeg.nii.gz")


def find_scans(data_root: Path, patient: str) -> list[Path]:
    pdir = data_root / patient
    if not pdir.is_dir():
        return []
    if (pdir / "PET.nii.gz").exists():
        return [pdir]
    return sorted(d for d in pdir.iterdir() if (d / "PET.nii.gz").exists())


def render(scan_dir: Path, label: str, out_png: Path) -> dict:
    pet, ct, seg = load(scan_dir)
    seg = seg > 0

    # coronal MIP: collapse the anterior-posterior axis (y) of a (z, y, x) volume
    pet_mip = pet.max(axis=1)
    ct_mip = ct.max(axis=1)
    seg_mip = seg.max(axis=1)

    ax_idx = (int(np.argmax(seg.reshape(seg.shape[0], -1).sum(1)))
              if seg.any() else ct.shape[0] // 2)

    stats = {
        "shape": tuple(int(v) for v in pet.shape),
        "ct_min": float(ct.min()), "ct_max": float(ct.max()), "ct_mean": float(ct.mean()),
        "ct_frac_air": float((ct <= -900).mean()),
        "pet_max": float(pet.max()),
        "lesion_voxels": int(seg.sum()),
    }

    # Per-lesion peak against the body's MEDIAN uptake, not mask-mean against
    # body-mean: a mean over the mask is dragged down by partial volume at every
    # lesion edge, and a bright bladder inflates the body mean, so a genuine
    # multi-lesion case can score barely above 1 while every lesion is plainly
    # on a hot spot. The per-lesion peak is what a reader actually looks at.
    body = (ct > -500) & ~seg
    if seg.any() and body.any():
        bg_median = float(np.median(pet[body])) or 1.0
        labels, n = ndi.label(seg)
        peaks = ndi.maximum(pet, labels, index=range(1, n + 1))
        ratios = sorted((float(p) / bg_median for p in np.atleast_1d(peaks)), reverse=True)
        stats["n_lesions"] = int(n)
        stats["lesion_peak_ratios"] = ratios
        stats["worst_lesion_ratio"] = ratios[-1] if ratios else float("nan")
        stats["best_lesion_ratio"] = ratios[0] if ratios else float("nan")
    else:
        stats["n_lesions"] = 0
        stats["lesion_peak_ratios"] = []
        stats["worst_lesion_ratio"] = stats["best_lesion_ratio"] = float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    # PET MIP, log-ish scaling so both the bright lesions and the body show
    vmax = np.percentile(pet_mip, 99.5) or 1
    axes[0].imshow(pet_mip, cmap="hot", origin="lower", vmin=0, vmax=vmax, aspect="auto")
    if seg_mip.any():
        axes[0].contour(seg_mip, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[0].set_title(f"PET coronal MIP + mask\nlesion voxels: {stats['lesion_voxels']}")

    axes[1].imshow(ct_mip, cmap="gray", origin="lower", vmin=-200, vmax=400, aspect="auto")
    axes[1].set_title(f"CT coronal MIP\nHU {stats['ct_min']:.0f}..{stats['ct_max']:.0f}, "
                      f"air {stats['ct_frac_air']:.0%}")

    axes[2].imshow(ct[ax_idx], cmap="gray", vmin=-200, vmax=400)
    if seg[ax_idx].any():
        axes[2].contour(seg[ax_idx], levels=[0.5], colors="cyan", linewidths=1.0)
    axes[2].set_title(f"CT axial slice {ax_idx}")

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    if stats["n_lesions"]:
        fig.suptitle(f"{label}   |   {stats['n_lesions']} lesion(s), peak/body-median "
                     f"{stats['best_lesion_ratio']:.0f}x..{stats['worst_lesion_ratio']:.0f}x")
    else:
        fig.suptitle(f"{label}   |   tumour-free")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("patients", nargs="+")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    for patient in args.patients:
        scans = find_scans(args.data_root, patient)
        if not scans:
            print(f"{patient}: no scans found under {args.data_root}")
            continue
        for scan_dir in scans:
            rel = scan_dir.relative_to(args.data_root)
            label = rel.as_posix()
            png = args.out / f"{label.replace('/', '__')}.png"
            s = render(scan_dir, label, png)
            verdict = []
            verdict.append("CT LOOKS BLANK" if s["ct_frac_air"] > 0.99 else "CT ok")
            if s["n_lesions"]:
                # every lesion should peak clearly above background; 2x on the
                # *weakest* one is a deliberately forgiving floor, since AutoPET
                # does contain genuinely faint lesions
                bad = [r for r in s["lesion_peak_ratios"] if r < 2]
                verdict.append(
                    f"{s['n_lesions']} lesion(s), peak/body-median "
                    f"{s['best_lesion_ratio']:.0f}x..{s['worst_lesion_ratio']:.0f}x"
                    + (f" -- {len(bad)} below 2x, INSPECT" if bad else " (all on hot spots)"))
            else:
                verdict.append("tumour-free")
            print(f"{label}: {', '.join(verdict)} | CT {s['ct_min']:.0f}..{s['ct_max']:.0f} HU "
                  f"| -> {png}")


if __name__ == "__main__":
    main()
