#!/usr/bin/env python3
"""Convert our MAE/MIM checkpoint into the form `nnUNetv2_train -pretrained_weights` wants.

The pretraining checkpoint stores the PlainConvUNet under a `nnunet.` prefix
(and `module.` too if it came from DataParallel), inside a training checkpoint
with optimizer and scheduler state. nnU-Net expects a plain
`{"network_weights": state_dict}` whose keys match the network it builds.

The output head is dropped on purpose. In pretraining `decoder.seg_layers`
reconstructed two image channels; in fine-tuning the same-shaped tensors predict
two classes. Identical shape, unrelated meaning -- copying them across would be
worse than the random initialisation nnU-Net would otherwise use, and nnU-Net
loads with strict=False so the fresh head simply stays fresh.

    python scripts/convert_mae_to_nnunet.py --checkpoint weights/nnunet_v2_mim_best.pth \\
        --out weights/nnunet_mae_pretrained_for_nnunet.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

STRIP_PREFIXES = ("module.", "nnunet.", "backbone.", "encoder_decoder.")
DROP_CONTAINING = ("seg_layers",)


def strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in STRIP_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    print(f"source checkpoint: {args.checkpoint}")
    print(f"  tensors: {len(sd)}")
    if isinstance(ck, dict):
        print(f"  epoch {ck.get('epoch')}, best_loss {ck.get('best_loss')}")

    out_sd, dropped = {}, []
    for k, v in sd.items():
        if any(s in k for s in DROP_CONTAINING):
            dropped.append(k)
            continue
        out_sd[strip_prefixes(k)] = v

    print(f"  kept {len(out_sd)} | dropped {len(dropped)} output-head tensors")
    n_params = sum(v.numel() for v in out_sd.values())
    print(f"  parameters kept: {n_params/1e6:.2f}M")

    first = next((k for k in out_sd if k.endswith("encoder.stages.0.0.convs.0.conv.weight")), None)
    if first:
        print(f"  first conv {first}: {tuple(out_sd[first].shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network_weights": out_sd}, args.out)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
