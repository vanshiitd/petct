#!/usr/bin/env python3
"""Derive nnUNetPlans_MAE.json from nnU-Net's own plan, with the pretrained network.

nnU-Net's planner adapts the network to the data: for anisotropic voxels it will
happily choose kernels like [1, 3, 3] and strides that downsample some axes
later than others. Our pretrained checkpoint was built with a fixed isotropic
configuration, so any such adaptation makes the weights unloadable.

This keeps everything the planner decided about the *data* -- spacing, patch
size, batch size, resampling, normalisation -- and replaces only the
architecture with the one the checkpoint was trained as:

    PlainConvUNet, 2 input channels, 6 stages, features 32/64/128/256/320/320,
    kernel [3,3,3] everywhere, strides [1,1,1] then [2,2,2] x5, 2 convs per
    stage in encoder and decoder, conv bias, InstanceNorm3d (eps 1e-5, affine),
    no dropout, LeakyReLU(0.01).

Five 2x downsamplings mean every patch dimension has to be divisible by 32, so
any that is not is rounded DOWN to the nearest multiple (rounding up would raise
memory beyond what the planner sized for).

    python scripts/make_mae_plans.py --preprocessed C:\\nnunet_preprocessed \\
        --dataset Dataset503_AutoPET_MAEprep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCH = {
    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
    "arch_kwargs": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * 6,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
        "n_conv_per_stage": [2] * 6,
        "n_conv_per_stage_decoder": [2] * 5,
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"negative_slope": 0.01, "inplace": True},
    },
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}
DIVISOR = 32  # 2 ** 5 downsamplings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preprocessed", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="3d_fullres")
    p.add_argument("--plans-name", default="nnUNetPlans_MAE")
    args = p.parse_args()

    ds_dir = args.preprocessed / args.dataset
    plans = json.loads((ds_dir / "nnUNetPlans.json").read_text())
    cfg = plans["configurations"][args.config]

    print(f"default plan for {args.config}:")
    print(f"  spacing      {cfg['spacing']}")
    print(f"  patch size   {cfg['patch_size']}")
    print(f"  batch size   {cfg['batch_size']}")
    print(f"  median shape {cfg['median_image_size_in_voxels']}")
    print(f"  normalisation {cfg['normalization_schemes']}")
    old = cfg["architecture"]["arch_kwargs"]
    print(f"  planned kernels {old['kernel_sizes']}")
    print(f"  planned strides {old['strides']}")
    print(f"  planned stages  {old['n_stages']}, features {old['features_per_stage']}")

    patch = list(cfg["patch_size"])
    adjusted = [max(DIVISOR, (v // DIVISOR) * DIVISOR) for v in patch]
    if adjusted != patch:
        print(f"\npatch size {patch} -> {adjusted} (each dimension rounded down to a "
              f"multiple of {DIVISOR}, for five 2x downsamplings)")
    else:
        print(f"\npatch size {patch} already divisible by {DIVISOR}")

    cfg["patch_size"] = adjusted
    cfg["architecture"] = json.loads(json.dumps(ARCH))
    plans["plans_name"] = args.plans_name

    # keep only the configuration we train, so nothing stale can be picked up
    plans["configurations"] = {args.config: cfg}

    out = ds_dir / f"{args.plans_name}.json"
    out.write_text(json.dumps(plans, indent=1))
    print(f"\nspacing kept at {cfg['spacing']} so nnU-Net does not resample the "
          f"already-prepared data")
    print(f"architecture replaced with the pretrained one "
          f"({ARCH['arch_kwargs']['n_stages']} stages, isotropic 3x3x3 kernels)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
