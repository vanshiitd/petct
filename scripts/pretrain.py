#!/usr/bin/env python3
"""Stage 1: MAE pretraining on unlabelled PET/CT volumes.

Examples:
    python scripts/pretrain.py --arch base
    python scripts/pretrain.py --arch nnunet --epochs 200 --batch-size 16
    python scripts/pretrain.py --arch large --device cuda:1 --data-root /data/PETCTfoundation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config, devices  # noqa: E402
from petct.config import ARCHS, get_arch  # noqa: E402
from petct.pretrain import run_pretraining  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ARCHS), help="backbone variant")
    p.add_argument("--data-root", type=Path, default=config.PATHS.pretrain_root,
                   help="root of the unlabelled .npy corpus")
    p.add_argument("--output-dir", type=Path, default=config.PATHS.weights_dir,
                   help="where checkpoints are written")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    p.add_argument("--epochs", type=int, default=config.PRETRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None, help="default: per-arch value from config")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=None, help="seed RNGs for reproducibility")
    p.add_argument("--no-data-parallel", action="store_true",
                   help="disable multi-GPU DataParallel even if several GPUs are visible")
    args = p.parse_args()

    if args.seed is not None:
        devices.seed_everything(args.seed)

    run_pretraining(
        arch=get_arch(args.arch),
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=devices.resolve_device(args.device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        data_parallel=not args.no_data_parallel,
    )


if __name__ == "__main__":
    main()
