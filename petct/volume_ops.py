"""Pure-numpy volume operations shared by every preprocessing script.

Deliberately free of torch/MONAI imports so the preprocessing scripts can run
on a machine with only numpy + scipy + SimpleITK installed.

These four functions were duplicated (with small, mostly cosmetic drift) across
autopetproc, spadeproc and DeepPSMA. This is the reconciled version.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def normalize_volume(volume: np.ndarray, method: str = "z-score") -> np.ndarray:
    """Normalise a 3D volume.

    z-score: (x - mean) / std   -- leaves 0.0 meaning "average intensity",
                                   which the MAE's zero-imputation relies on.
    min-max: scaled to [0, 1]
    """
    volume = volume.astype(np.float32)

    if method == "z-score":
        mu = float(volume.mean())
        sigma = float(volume.std())
        if sigma < 1e-8:
            return volume - mu
        return (volume - mu) / sigma

    if method == "min-max":
        v_min, v_max = float(volume.min()), float(volume.max())
        if (v_max - v_min) < 1e-8:
            return volume - v_min
        return (volume - v_min) / (v_max - v_min)

    raise ValueError("method must be 'z-score' or 'min-max'")


def get_body_bbox(
    ct_array: np.ndarray,
    threshold: float = -500,
    pad: int = 3,
    shape: tuple[int, int, int] | None = None,
):
    """Tight 3D bounding box around the patient's body, from CT.

    Thresholding at -500 HU separates tissue from air; taking only the largest
    connected component then discards the scanner bed and any ring artefacts at
    the edge of the field of view.

    ct_array is expected in (z, y, x) order. `shape` bounds the returned slices
    (defaults to ct_array's own shape) -- pass the PET shape when cropping a
    resampled pair whose dimensions should agree.

    Returns a (slice, slice, slice) tuple, or None if no body was found.
    """
    body_mask = ct_array > threshold
    labels, num_features = ndi.label(body_mask)
    if num_features == 0:
        return None

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    clean_body_mask = labels == counts.argmax()

    z_idx, y_idx, x_idx = np.where(clean_body_mask)
    if len(z_idx) == 0:
        return None

    bound = shape if shape is not None else ct_array.shape
    return (
        slice(max(0, z_idx.min() - pad), min(bound[0], z_idx.max() + pad + 1)),
        slice(max(0, y_idx.min() - pad), min(bound[1], y_idx.max() + pad + 1)),
        slice(max(0, x_idx.min() - pad), min(bound[2], x_idx.max() + pad + 1)),
    )


def resample_to_spacing(
    array: np.ndarray,
    orig_spacing_zyx: np.ndarray,
    target_spacing_zyx: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Resample a volume to a target voxel spacing.

    order=1 (trilinear) for images, order=0 (nearest) for label masks -- using
    interpolation on a label map would invent non-existent class values.
    """
    zoom = np.asarray(orig_spacing_zyx, dtype=float) / np.asarray(target_spacing_zyx, dtype=float)
    return ndi.zoom(array, zoom, order=order)


def resample_to_shape(array: np.ndarray, target_shape, order: int = 1) -> np.ndarray:
    """Resample a volume so its shape matches `target_shape` exactly."""
    zoom = np.asarray(target_shape, dtype=float) / np.asarray(array.shape, dtype=float)
    return ndi.zoom(array, zoom, order=order)
