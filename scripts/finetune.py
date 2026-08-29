#!/usr/bin/env python3
"""Stage 2: supervised fine-tuning for tumour segmentation.

Replaces all 18 of the {10,30,100}%training[-mae].ipynb notebooks across the
four backbone folders. Those differed only in train_fraction, use_foundation,
and the backbone's feature_size/depths -- all now flags.

Examples:
    # foundation-initialised, 100% of labels, base backbone
    python scripts/finetune.py --arch base --fraction 1.0 --init foundation

    # the scratch comparison arm at 10% labels
    python scripts/finetune.py --arch base --fraction 0.1 --init scratch

    # quick local smoke test on the 20-patient sample set
    python scripts/finetune.py --arch base --fraction 1.0 --init foundation \\
        --split sample --epochs 2 --batch-size 1 --val-interval 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petct import config, devices  # noqa: E402
from petct.config import ARCHS, SPLITS, get_arch  # noqa: E402
from petct.finetune import run_finetuning  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ARCHS))
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the labelled training pool to use (e.g. 0.1, 0.3, 1.0)")
    p.add_argument("--init", choices=["foundation", "scratch"], default="foundation",
                   help="foundation = load pretrained weights; scratch = random init")
    p.add_argument("--foundation-ckpt", type=Path, default=None,
                   help="explicit pretrained checkpoint (default: <weights-dir>/<arch ckpt>)")
    p.add_argument("--data-root", type=Path, default=config.PATHS.autopet_root,
                   help="root of the labelled AutoPET data")
    p.add_argument("--output-dir", type=Path, default=config.PATHS.output_dir)
    p.add_argument("--split", choices=sorted(SPLITS), default="full",
                   help="'full' = 200-patient test set; 'sample' = small local datasets")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    p.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sw-batch-size", type=int, default=1,
                   help="sliding-window inference batch; lower this if evaluation OOMs")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val-interval", type=int, default=config.VAL_INTERVAL)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not 0 < args.fraction <= 1.0:
        raise SystemExit(f"--fraction must be in (0, 1]; got {args.fraction}")

    if args.seed is not None:
        devices.seed_everything(args.seed)

    run_finetuning(
        arch=get_arch(args.arch),
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=devices.resolve_device(args.device),
        train_fraction=args.fraction,
        use_foundation=(args.init == "foundation"),
        foundation_ckpt=args.foundation_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sw_batch_size=args.sw_batch_size,
        lr=args.lr,
        val_interval=args.val_interval,
        split_name=args.split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
