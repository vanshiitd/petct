"""Model construction, MAE/MIM wrappers, and pretrained-weight transfer."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .config import ArchSpec


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------
def build_swin_unetr(arch: ArchSpec, out_channels: int) -> nn.Module:
    """SwinUNETR v2 backbone.

    `img_size` was a required constructor argument in older MONAI and was
    REMOVED in newer versions (>=1.5), which infer spatial size at runtime.
    Passing it unconditionally raises a TypeError on current MONAI, so we
    inspect the signature rather than pinning a version.
    """
    from monai.networks.nets import SwinUNETR

    kwargs = dict(
        in_channels=config.IN_CHANNELS,
        out_channels=out_channels,
        feature_size=arch.feature_size,
        depths=tuple(arch.depths),
        num_heads=tuple(arch.num_heads),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.05,
        normalize=True,
        use_checkpoint=True,
        spatial_dims=3,
        downsample="mergingv2",
        use_v2=True,
    )
    if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
        kwargs["img_size"] = config.ROI_SIZE
    return SwinUNETR(**kwargs)


def build_nnunet(arch: ArchSpec, out_channels: int) -> nn.Module:
    """nnU-Net v2 PlainConvUNet backbone.

    These stage parameters replicate nnU-Net v2's 3d_fullres heuristics for a
    96x128x128 input: stage 0 at full resolution, then five 2x downsamples
    reaching a 3x4x4 bottleneck.

    Note this is the nnU-Net *network*, hand-configured. It is not the full
    self-configuring nnU-Net pipeline (which plans these values from a dataset
    fingerprint). See CONVERSION_NOTES.md.
    """
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from dynamic_network_architectures.initialization.weight_init import InitWeights_He

    model = PlainConvUNet(
        input_channels=config.IN_CHANNELS,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],  # 320 = nnU-Net's 3D cap
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_conv_per_stage=[2] * 6,
        num_classes=out_channels,
        n_conv_per_stage_decoder=[2] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=False,  # only the full-resolution head, to match SwinUNETR eval
    )
    model.apply(InitWeights_He(1e-2))
    return model


def build_backbone(arch: ArchSpec, out_channels: int) -> nn.Module:
    if arch.family == "swin":
        return build_swin_unetr(arch, out_channels)
    if arch.family == "nnunet":
        return build_nnunet(arch, out_channels)
    raise ValueError(f"Unknown family {arch.family!r}")


# ---------------------------------------------------------------------------
# Masked autoencoding wrapper
# ---------------------------------------------------------------------------
class MaskedAutoencoder(nn.Module):
    """Wraps a segmentation backbone into a masked-reconstruction pretrainer.

    Three design choices, all inherited from the source method:

    1. PET and CT are masked INDEPENDENTLY. The mask is drawn per-channel, so
       at many locations one modality is hidden while the other is visible --
       forcing the network to reconstruct anatomy from metabolism and vice
       versa, which is the cross-modal signal the whole approach is after.
    2. Masked voxels are set to 0.0 rather than a learned mask token. Because
       each channel was z-scored, 0.0 IS the dataset mean, so this fills holes
       with the statistically expected value and avoids the blocky seams a
       learned token can produce.
    3. Loss is weighted 1.0 on masked regions, 0.2 on visible ones. The visible
       term is a stabiliser: with U-Net skip connections, visible information
       can flow almost straight to the output, so leaving it unscored lets the
       network garble it for free.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_patch_size=config.MASK_PATCH_SIZE,
        mask_ratio: float = config.MASK_RATIO,
        unmasked_weight: float = config.UNMASKED_LOSS_WEIGHT,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_patch_size = tuple(mask_patch_size)
        self.mask_ratio = mask_ratio
        self.unmasked_weight = unmasked_weight

    def forward(self, x: torch.Tensor):
        b, c, z, y, xx = x.shape
        pz, py, px = self.mask_patch_size
        if z % pz or y % py or xx % px:
            raise ValueError(
                f"Input {(z, y, xx)} is not divisible by mask patch size "
                f"{self.mask_patch_size}."
            )

        # One independent mask per (sample, channel) over the block grid.
        noise = torch.rand(b, c, z // pz, y // py, xx // px, device=x.device)
        mask = (noise < self.mask_ratio).float()
        mask = (
            mask.repeat_interleave(pz, dim=2)
            .repeat_interleave(py, dim=3)
            .repeat_interleave(px, dim=4)
        )

        x_masked = x * (1 - mask)  # zero == post-normalisation mean
        x_rec = self.backbone(x_masked)

        per_voxel = F.mse_loss(x_rec, x, reduction="none")
        loss_masked = (per_voxel * mask).sum() / (mask.sum() + 1e-8)
        loss_visible = (per_voxel * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        loss = loss_masked + self.unmasked_weight * loss_visible
        return loss, x_rec


def build_pretrain_model(arch: ArchSpec) -> MaskedAutoencoder:
    """Backbone configured for reconstruction: out_channels == in_channels."""
    backbone = build_backbone(arch, out_channels=config.IN_CHANNELS)
    return MaskedAutoencoder(backbone)


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------
def _strip_module_prefix(state: dict) -> dict:
    """Undo the 'module.' prefix added by nn.DataParallel when saving."""
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state.items()}
    return state


def load_foundation_weights(
    model: nn.Module, ckpt_path: Path, arch: ArchSpec, verbose: bool = True
) -> nn.Module:
    """Transfer pretrained backbone weights into a segmentation model.

    The pretraining checkpoint stores the backbone nested inside the MAE
    wrapper, so keys carry a prefix that must be stripped to line up with a
    bare backbone. Keys matching arch.skip_on_transfer are dropped (the nnU-Net
    output head reconstructs images during pretraining and classifies during
    fine-tuning -- same shape, incompatible meaning).

    strict=False is required, but it also means a prefix mismatch would
    silently transfer NOTHING and leave you training from random weights. So we
    check and fail loudly instead.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Foundation checkpoint not found: {ckpt_path}\n"
            f"Pass --foundation-ckpt, or run pretraining first, or use --init scratch."
        )

    if verbose:
        print(f"==> Loading foundation weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = _strip_module_prefix(state)

    # The wrapper attribute is "backbone." here; historical checkpoints used
    # "encoder_decoder." (swin) or "nnunet." (nnU-Net). Accept any of them.
    prefixes = ("backbone.", arch.weight_prefix)
    transferred = {}
    for k, v in state.items():
        for p in prefixes:
            if k.startswith(p):
                new_key = k[len(p):]
                if any(s in new_key for s in arch.skip_on_transfer):
                    break
                transferred[new_key] = v
                break

    if not transferred:
        raise SystemExit(
            f"Checkpoint {ckpt_path} contained no keys with prefix "
            f"{prefixes}. Refusing to continue -- the model would silently "
            f"train from random weights."
        )

    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if verbose:
        print(
            f"Weight transfer: {len(transferred)} tensors loaded | "
            f"missing: {len(missing)} | unexpected: {len(unexpected)}"
        )
        if arch.skip_on_transfer and missing:
            print(f"  (missing keys are expected here: {arch.skip_on_transfer} head is re-initialised)")
    return model


def build_segmentation_model(
    arch: ArchSpec,
    device: torch.device,
    use_foundation: bool,
    foundation_ckpt: Path | None = None,
) -> nn.Module:
    """Build the fine-tuning model, optionally initialised from pretraining.

    IMPORTANT: this returns a fully-initialised model, and the optimizer must
    be created from its parameters AFTER this call. Building a model, binding
    an optimizer to it, and only then replacing the model leaves the optimizer
    updating an orphaned copy -- the network then silently never learns. See
    DESIGN_NOTES.md.
    """
    model = build_backbone(arch, out_channels=config.NUM_CLASSES)
    if use_foundation:
        ckpt = Path(foundation_ckpt or (config.PATHS.weights_dir / arch.pretrain_ckpt))
        model = load_foundation_weights(model, ckpt, arch)
    else:
        print("==> Starting from random weights (scratch).")
    return model.to(device)
