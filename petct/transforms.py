"""MONAI transforms for the segmentation (fine-tuning) pipeline."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from monai.transforms import MapTransform

from . import config
from .volume_ops import get_body_bbox, normalize_volume, resample_to_shape, resample_to_spacing


class AutoPETPreprocessd(MapTransform):
    """Load a PET/CT/mask triplet from disk and turn it into model input.

    Reads the three NIfTI files named in the data dict, then applies the same
    chain the pretraining corpus went through, so a fine-tuned model sees data
    distributed exactly like what it was pretrained on:

        resample to common spacing -> crop to body -> z-score -> stack channels

    Produces:
        d["image"] : float32 (2, Z, Y, X)  -- channel 0 PET, channel 1 CT
        d["label"] : float32 (1, Z, Y, X)  -- binary tumour mask
    """

    def __init__(self, keys, target_spacing_zyx=config.TARGET_SPACING_ZYX):
        super().__init__(keys)
        self.target_spacing_zyx = np.asarray(target_spacing_zyx, dtype=float)

    def __call__(self, data):
        d = dict(data)

        pet_img = sitk.ReadImage(d["pet_path"])
        ct_img = sitk.ReadImage(d["ct_path"])
        seg_img = sitk.ReadImage(d["seg_path"])

        # SimpleITK reports spacing as (x, y, z); the arrays come out (z, y, x).
        spacing_xyz = np.round(pet_img.GetSpacing()).astype(float)
        orig_spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])

        pet = sitk.GetArrayFromImage(pet_img).astype(np.float32)
        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        seg = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
        seg = (seg > 0).astype(np.uint8)  # guard against stray label values

        pet_r = resample_to_spacing(pet, orig_spacing_zyx, self.target_spacing_zyx, order=1)
        ct_r = resample_to_shape(ct, pet_r.shape, order=1)
        seg_r = resample_to_shape(seg, pet_r.shape, order=0)  # nearest for labels

        bbox = get_body_bbox(
            ct_r,
            threshold=config.CT_BODY_THRESHOLD,
            pad=config.CROP_PAD,
            shape=pet_r.shape,
        )
        if bbox is not None:
            pet_r, ct_r, seg_r = pet_r[bbox], ct_r[bbox], seg_r[bbox]

        image = np.stack([normalize_volume(pet_r), normalize_volume(ct_r)], axis=0)

        d["image"] = image.astype(np.float32)
        d["label"] = np.expand_dims(seg_r, axis=0).astype(np.float32)
        return d
