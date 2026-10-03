"""Regression test: the lesion mask must not be mirrored when placed on the PET grid.

`dicom_to_nifti.convert_patient` used to call `CopyInformation(pet)` on the
segmentation whenever it had the same size as the PET, which stamps the PET's
geometry onto the mask rather than moving the mask onto the PET's grid. Equal
size does not mean equal geometry: for 317 of this collection's 501
tumour-positive scans the SEG's orientation differs from the PET's, and the
shortcut mirrored the lesion left-right.

Nothing downstream caught it -- the mask kept its shape, its size and its voxel
count -- and the models trained on it simply could not fit the labels. So the
test here is not "does it run" but "does the mask land on the tracer".

Two checks:

  * a synthetic case built so the SEG's direction disagrees with the PET's. The
    old shortcut mirrors it; resampling places it correctly.
  * the real scan PETCT_3b1c9155f5, whose mask was mirrored by the old code,
    skipped when the data is not present.

    pytest tests/test_seg_orientation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_V2 = Path(r"C:\data\autopet_nifti_v2")
MIRRORED_CASE = "PETCT_3b1c9155f5"


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * int(np.logical_and(a, b).sum()) / denom


def test_resampling_beats_copyinformation_when_direction_differs():
    """A SEG whose direction disagrees with the PET must still land correctly."""
    size = (16, 16, 8)
    pet = sitk.Image(size, sitk.sitkFloat32)
    pet.SetSpacing((2.0, 2.0, 3.0))
    pet.SetOrigin((0.0, 0.0, 0.0))
    pet.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # a blob on one side only, so a mirror is unmistakable
    arr = np.zeros(size[::-1], dtype=np.uint8)
    arr[2:5, 2:6, 2:5] = 1
    seg = sitk.GetImageFromArray(arr)
    seg.SetSpacing(pet.GetSpacing())
    # y axis runs the other way, and the origin moves to the far edge so the
    # blob still occupies the same physical place
    seg.SetDirection((1, 0, 0, 0, -1, 0, 0, 0, 1))
    seg.SetOrigin((0.0, (size[1] - 1) * 2.0, 0.0))

    resampled = sitk.GetArrayFromImage(
        sitk.Resample(seg, pet, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ) > 0

    copied = sitk.Image(seg)
    copied.CopyInformation(pet)                      # the old behaviour
    copied_arr = sitk.GetArrayFromImage(copied) > 0

    # the mask's physical place is where resampling puts it
    assert resampled.sum() > 0, "resampling lost the mask entirely"
    assert _dice(resampled, copied_arr) < 0.5, (
        "CopyInformation happened to agree here, so this case does not exercise the bug"
    )
    # and the two differ by exactly a left-right mirror
    assert _dice(resampled, copied_arr[:, ::-1, :]) == pytest.approx(1.0), (
        "the disagreement should be a pure y mirror"
    )


@pytest.mark.skipif(not (DATA_V2 / MIRRORED_CASE / "tumorSeg.nii.gz").exists(),
                    reason="converted data not present")
def test_known_mirrored_case_sits_on_tracer():
    """The real scan whose mask the old code mirrored.

    A lesion label must sit on elevated FDG uptake. Before the fix this mask
    scored 0.68x the body's median -- i.e. it was on tissue dimmer than
    background, the signature of a mirrored mask.
    """
    d = DATA_V2 / MIRRORED_CASE
    pet = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "PET.nii.gz")))
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "CT_resample.nii.gz")))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "tumorSeg.nii.gz"))) > 0

    assert seg.any(), "this case is tumour-positive and must have a mask"
    body = ct > -500
    inside = float(np.median(pet[seg]))
    outside = float(np.median(pet[body & ~seg]))
    ratio = inside / max(outside, 1e-6)

    assert ratio > 2.0, (
        f"lesion mask sits on tissue at {ratio:.2f}x the body median; a correctly "
        f"placed mask is well above 1, a mirrored one lands near or below it"
    )
    # and the mirror must now be the worse option
    flipped = seg[:, ::-1, :]
    ratio_flipped = float(np.median(pet[flipped])) / max(outside, 1e-6)
    assert ratio > ratio_flipped, (
        f"flipping the mask improves uptake ({ratio_flipped:.2f}x vs {ratio:.2f}x), "
        f"which means it is still mirrored"
    )
