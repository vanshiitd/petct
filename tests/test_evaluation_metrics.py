"""The new Experiment-1 metrics, checked against masks whose answers are known.

Every case here is small enough to work out by hand, because a metric that is
silently wrong is worse than one that is missing: it produces a plausible number
that nothing downstream questions. These are the arithmetic checks the mirrored
masks never had.

    pytest tests/test_evaluation_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_predictions import (  # noqa: E402
    assd_mm, component_counts, dice, hd95_mm, iou, fp_fn_volumes_ml,
)

ISO = (1.0, 1.0, 1.0)


def block(shape, sl):
    a = np.zeros(shape, dtype=bool)
    a[sl] = True
    return a


# --- IoU ------------------------------------------------------------------

def test_iou_identical_is_one():
    a = block((10, 10, 10), np.s_[2:5, 2:5, 2:5])
    assert iou(a, a) == pytest.approx(1.0)
    assert dice(a, a) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = block((10, 10, 10), np.s_[0:2, 0:2, 0:2])
    b = block((10, 10, 10), np.s_[7:9, 7:9, 7:9])
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_half_overlap_known_value():
    """Two 2x2x2 cubes sharing exactly half their volume.

    intersection 4, union 12 -> IoU 1/3, Dice 2*4/(8+8) = 1/2.
    """
    a = block((10, 10, 10), np.s_[0:2, 0:2, 0:2])
    b = block((10, 10, 10), np.s_[0:2, 0:2, 1:3])
    assert int(np.logical_and(a, b).sum()) == 4
    assert int(np.logical_or(a, b).sum()) == 12
    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a, b) == pytest.approx(0.5)


def test_iou_undefined_when_both_empty():
    z = np.zeros((4, 4, 4), dtype=bool)
    assert np.isnan(iou(z, z))
    assert np.isnan(dice(z, z))


# --- ASSD and HD95 --------------------------------------------------------

def test_assd_identical_is_zero():
    a = block((12, 12, 12), np.s_[3:8, 3:8, 3:8])
    assert assd_mm(a, a, ISO) == pytest.approx(0.0)
    assert hd95_mm(a, a, ISO) == pytest.approx(0.0)


def test_assd_undefined_against_empty():
    a = block((8, 8, 8), np.s_[1:3, 1:3, 1:3])
    z = np.zeros((8, 8, 8), dtype=bool)
    assert np.isnan(assd_mm(a, z, ISO))
    assert np.isnan(hd95_mm(a, z, ISO))


def test_assd_separated_slabs_known_value():
    """Two 4-thick slabs with a 3-voxel gap, worked out voxel by voxel.

    Grid 20x6x6, 1 mm isotropic. Slab a spans z in [0, 4), slab b spans
    z in [7, 11); both fill x and y. `surface_voxels` erodes with
    border_value=0, so the array edge counts as outside and a slab's surface is
    every voxel except its interior z in {1, 2}, x,y in [1, 5):

        z=0  36 voxels   z=1  20   z=2  20   z=3  36        (112 in total)

    Nearest b voxel is at z=7 directly above, so the directed distances are
    7, 6, 5 and 4 mm:

        (36*7 + 20*6 + 20*5 + 36*4) / 112 = 616 / 112 = 5.5

    and b -> a is the mirror image, also 5.5, so ASSD is 5.5 mm.
    """
    shape = (20, 6, 6)
    a = block(shape, np.s_[0:4, :, :])
    b = block(shape, np.s_[7:11, :, :])
    assert assd_mm(a, b, ISO) == pytest.approx(5.5, abs=1e-9)
    # and it is symmetric in its arguments
    assert assd_mm(b, a, ISO) == pytest.approx(5.5, abs=1e-9)


def test_assd_anisotropic_spacing_scales():
    """The same voxel displacement costs more when voxels are larger."""
    shape = (20, 6, 6)
    a = block(shape, np.s_[0:4, :, :])
    b = block(shape, np.s_[7:11, :, :])
    iso = assd_mm(a, b, (1.0, 1.0, 1.0))
    stretched = assd_mm(a, b, (3.0, 1.0, 1.0))
    assert stretched == pytest.approx(3.0 * iso, rel=1e-6)


def test_hd95_is_at_least_assd():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random((14, 14, 14)) > 0.7
        b = rng.random((14, 14, 14)) > 0.7
        h, s = hd95_mm(a, b, ISO), assd_mm(a, b, ISO)
        assert h >= s - 1e-9, f"HD95 {h} below ASSD {s}"


# --- connectivity ---------------------------------------------------------

def test_component_counts_three_and_two():
    shape = (20, 20, 20)
    pred = np.zeros(shape, dtype=bool)
    pred[1:3, 1:3, 1:3] = True
    pred[10:12, 10:12, 10:12] = True
    pred[16:18, 2:4, 2:4] = True
    gt = np.zeros(shape, dtype=bool)
    gt[1:3, 1:3, 1:3] = True
    gt[6:8, 15:17, 15:17] = True
    n_pred, n_gt = component_counts(pred, gt)
    assert (n_pred, n_gt) == (3, 2)


def test_lesion_sensitivity_and_fp_lesions():
    """One of two true lesions found; two of three predictions are spurious.

    Each blob is 2x2x2 = 8 voxels, and with 1 mm isotropic voxels 8 voxels is
    0.008 mL, so the volumes are known exactly too.
    """
    shape = (20, 20, 20)
    pred = np.zeros(shape, dtype=bool)
    pred[1:3, 1:3, 1:3] = True          # hits true lesion 1
    pred[10:12, 10:12, 10:12] = True    # spurious
    pred[16:18, 2:4, 2:4] = True        # spurious
    gt = np.zeros(shape, dtype=bool)
    gt[1:3, 1:3, 1:3] = True            # found
    gt[6:8, 15:17, 15:17] = True        # missed

    fp_ml, fn_ml, n_fp, n_fn = fp_fn_volumes_ml(pred, gt, ISO)
    n_pred, n_gt = component_counts(pred, gt)
    assert (n_fp, n_fn) == (2, 1)
    assert n_gt - n_fn == 1                      # one true lesion hit
    assert (n_gt - n_fn) / n_gt == pytest.approx(0.5)
    assert fp_ml == pytest.approx(2 * 8 / 1000.0)
    assert fn_ml == pytest.approx(1 * 8 / 1000.0)


# --- volume error ---------------------------------------------------------

def test_volume_error_known_values():
    """pred 27 voxels against true 8, at 2x1x1 mm: 2 mm^3 per voxel."""
    shape = (12, 12, 12)
    pred = block(shape, np.s_[0:3, 0:3, 0:3])
    gt = block(shape, np.s_[0:2, 0:2, 0:2])
    spacing = (2.0, 1.0, 1.0)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    pred_ml = pred.sum() * voxel_ml
    gt_ml = gt.sum() * voxel_ml
    assert pred.sum() == 27 and gt.sum() == 8
    assert pred_ml == pytest.approx(27 * 0.002)
    assert gt_ml == pytest.approx(8 * 0.002)
    assert pred_ml - gt_ml == pytest.approx(19 * 0.002)
    assert (pred_ml - gt_ml) / gt_ml == pytest.approx(19 / 8)
