#!/usr/bin/env python3
"""
View a patient's PET, CT, and tumor segmentation from sample_data/.

Usage:
    python view_scan.py                        # list available patients
    python view_scan.py PETCT_1bb48bfb40        # view that patient (auto-picks the
                                                 # axial slice with the most tumor)
    python view_scan.py PETCT_1bb48bfb40 --slice 120
    python view_scan.py PETCT_1bb48bfb40 --out my_view.png

Saves a PNG (doesn't try to pop up a window, since this is meant to be run
headlessly) showing three panels: CT, PET, and CT with the tumor mask overlaid.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

SAMPLE_DATA = Path(__file__).resolve().parent / "sample_data"


def load_patient(patient_dir):
    pet = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "PET.nii.gz")))
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "CT_resample.nii.gz")))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "tumorSeg.nii.gz")))
    return pet, ct, seg  # each shape (Z, Y, X)


def pick_slice(seg):
    tumor_per_slice = (seg > 0).reshape(seg.shape[0], -1).sum(axis=1)
    if tumor_per_slice.max() == 0:
        return seg.shape[0] // 2  # no tumor in this patient: just show the middle slice
    return int(np.argmax(tumor_per_slice))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patient_id", nargs="?", help="folder name under sample_data/, e.g. PETCT_1bb48bfb40")
    parser.add_argument("--slice", type=int, default=None, help="axial slice index (default: slice with most tumor)")
    parser.add_argument("--out", default=None, help="output PNG path (default: <patient_id>_slice<N>.png)")
    args = parser.parse_args()

    if args.patient_id is None:
        patients = sorted(p.name for p in SAMPLE_DATA.iterdir() if p.is_dir())
        print(f"Available patients ({len(patients)}) in {SAMPLE_DATA}:")
        for p in patients:
            print(" ", p)
        print(f"\nRun again with one, e.g.:\n  python {Path(__file__).name} {patients[0]}")
        return

    patient_dir = SAMPLE_DATA / args.patient_id
    if not patient_dir.exists():
        sys.exit(f"No such patient folder: {patient_dir}")

    pet, ct, seg = load_patient(patient_dir)
    z = args.slice if args.slice is not None else pick_slice(seg)
    z = max(0, min(z, pet.shape[0] - 1))

    ct_slice, pet_slice, seg_slice = ct[z], pet[z], seg[z]
    tumor_total = int((seg > 0).sum())
    tumor_here = int((seg_slice > 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"{args.patient_id}  —  axial slice {z}/{pet.shape[0]-1}  "
        f"(tumor voxels: {tumor_here} in this slice, {tumor_total} in the whole scan)",
        fontsize=11,
    )

    ct_lo, ct_hi = np.percentile(ct_slice, [1, 99])
    axes[0].imshow(ct_slice, cmap="gray", vmin=ct_lo, vmax=ct_hi)
    axes[0].set_title("CT")
    axes[0].axis("off")

    pet_lo, pet_hi = np.percentile(pet, [1, 99.5])
    axes[1].imshow(pet_slice, cmap="hot", vmin=pet_lo, vmax=pet_hi)
    axes[1].set_title("PET")
    axes[1].axis("off")

    axes[2].imshow(ct_slice, cmap="gray", vmin=ct_lo, vmax=ct_hi)
    masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[2].imshow(masked, cmap="autumn", alpha=0.6)
    axes[2].set_title("CT + tumor mask")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = args.out or f"{args.patient_id}_slice{z}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
