#!/usr/bin/env python3
"""Plot EMA pseudo-Dice against epoch for two or more nnU-Net runs.

nnU-Net logs the EMA only when it improves ("New best EMA pseudo Dice"), which
is a monotone trace and hides every dip. The per-epoch raw pseudo-Dice *is*
logged every epoch, so the EMA is recomputed here with nnU-Net's own rule,

    ema[0] = d[0];  ema[i] = 0.9 * ema[i-1] + 0.1 * d[i]

(`nnunet_logger.py`), and then checked against the "New best EMA" lines the run
actually printed. If the two disagree the plot is wrong and the script says so
rather than drawing it.

    python scripts/plot_training_curves.py --out reports/curves.png \\
        --run "MAE pretrained" D:\\data\\reports\\nnunet_505_fold0.log \\
        --run "from scratch"   D:\\data\\reports\\nnunet_505_scratch_fold0.log
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DICE = re.compile(r"Pseudo dice \[np\.float32\(([-\d.eE]+)\)\]")
BEST = re.compile(r"New best EMA pseudo Dice: ([\d.eE+-]+)")


def parse(path: Path) -> tuple[list[float], list[float], float]:
    text = path.read_text(errors="replace")
    dice = [float(m) for m in DICE.findall(text)]
    if not dice:
        raise SystemExit(f"no per-epoch pseudo dice found in {path}")
    ema, cur = [], None
    for d in dice:
        cur = d if cur is None else 0.9 * cur + 0.1 * d
        ema.append(cur)
    best_logged = max((float(m) for m in BEST.findall(text)), default=float("nan"))
    return dice, ema, best_logged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs=2, action="append", required=True,
                   metavar=("LABEL", "LOG"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="nnU-Net on Dataset505: EMA pseudo-Dice")
    args = p.parse_args()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    ok = True
    n_epochs = 0

    for i, (label, log) in enumerate(args.run):
        dice, ema, best_logged = parse(Path(log))
        c = colours[i % len(colours)]
        ax.plot(range(len(dice)), dice, color=c, alpha=0.18, lw=0.8)
        ax.plot(range(len(ema)), ema, color=c, lw=1.8,
                label=f"{label} (best {max(ema):.4f})")
        ax2.plot(range(len(ema)), ema, color=c, lw=1.8, label=label)

        n_epochs = max(n_epochs, len(dice))
        agree = abs(max(ema) - best_logged) < 5e-4
        ok &= agree
        print(f"{label:18s} epochs {len(dice):4d} | recomputed best EMA {max(ema):.4f} "
              f"| logged {best_logged:.4f} | {'OK' if agree else 'MISMATCH'}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("pseudo-Dice (validation patches)")
    ax.set_title(args.title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax2.set_xlim(0, min(100, n_epochs))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("EMA pseudo-Dice")
    ax2.set_title("first 100 epochs")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    if not ok:
        raise SystemExit("recomputed EMA disagrees with the logged best; not writing a plot")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
