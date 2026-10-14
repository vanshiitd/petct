#!/usr/bin/env python3
"""Widen a pretrained checkpoint's first convolution from 2 input channels to 4.

The refinement network (plan strategy 7) takes PET, CT, the M0 probability and
U_stab. The MAE weights were learned on PET and CT alone, so the first
convolution has to grow two input slots. They are filled with **zeros**, which
means the widened network computes exactly what the 2-channel one computed on
the same PET and CT: training starts as the base model and has to earn any use
it makes of the two new channels. Random values there would instead inject
noise into every feature map at step 0 and throw away the pretraining.

Nothing else changes. nnU-Net loads with strict=False, so the segmentation head
stays freshly initialised exactly as it would for a 2-channel run.

    python scripts/expand_input_channels.py \\
        --checkpoint weights/nnunet_mae_pretrained_for_nnunet.pth \\
        --out weights/nnunet_mae_pretrained_4ch.pth --in-channels 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--in-channels", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["network_weights"] if "network_weights" in ck else ck
    cands = [k for k, v in sd.items()
             if k.endswith("weight") and getattr(v, "ndim", 0) == 5]
    if not cands:
        raise SystemExit("no 5-D conv weight found; is this a 3-D U-Net?")
    in_c = min(sd[k].shape[1] for k in cands)
    if in_c > args.in_channels:
        raise SystemExit(f"checkpoint has {in_c} input channels, cannot shrink "
                         f"to {args.in_channels}")
    # The stem convolution appears under more than one key: dynamic-network-
    # architectures registers it as `.conv` and again inside `.all_modules`,
    # two names for one tensor. Widening only the first leaves the other at the
    # old width and nnU-Net refuses the checkpoint, so every alias is widened.
    # No other convolution in this network has so few input channels, so
    # "shape[1] == in_c" selects exactly the stem's aliases.
    firsts = [k for k in cands if sd[k].shape[1] == in_c]
    if in_c == args.in_channels:
        print(f"already {in_c} input channels; nothing to do")
    else:
        print(f"stem convolution appears under {len(firsts)} key(s):")
        for k in firsts:
            w = sd[k]
            new = torch.zeros((w.shape[0], args.in_channels, *w.shape[2:]),
                              dtype=w.dtype)
            new[:, :in_c] = w
            sd[k] = new
            print(f"  {k}: {tuple(w.shape)} -> {tuple(new.shape)}, "
                  f"norm {w.norm():.6f} -> {new.norm():.6f}")
        print(f"channels {in_c}..{args.in_channels-1} zero-filled, so the "
              f"network starts as the {in_c}-channel model")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network_weights": sd}, args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
