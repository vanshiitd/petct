"""Stage 2: supervised fine-tuning for tumour segmentation, plus evaluation."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.optim as optim
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from . import config, devices
from .config import ArchSpec
from .data import get_segmentation_dataloaders
from .models import build_segmentation_model


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    sw_batch_size: int = 1,
    desc: str = "eval",
) -> tuple[float, float]:
    """Run sliding-window inference over full volumes; return (Dice, HD95).

    Training used 96x128x128 crops, but a real scan is far larger than GPU
    memory allows in one pass. A window walks the volume at 50% overlap and the
    overlapping predictions are averaged, which also smooths the unreliable
    predictions near each window's edge.
    """
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=config.NUM_CLASSES)])

    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with devices.amp_autocast(device):
            logits = sliding_window_inference(
                inputs=images,
                roi_size=config.ROI_SIZE,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=config.SLIDING_WINDOW_OVERLAP,
            )

        preds = [post_pred(i) for i in logits]
        targets = [post_label(i) for i in labels]

        dice_metric(y_pred=preds, y=targets)
        try:
            hd95_metric(y_pred=preds, y=targets)
        except Exception:
            # HD95 is undefined when a case has no predicted or no true
            # foreground; skip those rather than aborting evaluation.
            pass

        del images, labels, logits, preds, targets
        devices.empty_cache(device)

    mean_dice = float(dice_metric.aggregate().item())
    try:
        mean_hd95 = float(hd95_metric.aggregate().item())
    except Exception:
        mean_hd95 = float("inf")
    dice_metric.reset()
    hd95_metric.reset()
    return mean_dice, mean_hd95


def run_finetuning(
    arch: ArchSpec,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    train_fraction: float = 1.0,
    use_foundation: bool = True,
    foundation_ckpt: Path | None = None,
    epochs: int = config.FINETUNE_EPOCHS,
    batch_size: int = 4,
    sw_batch_size: int = 1,
    lr: float | None = None,
    val_interval: int = config.VAL_INTERVAL,
    split_name: str = "full",
    num_workers: int = 4,
) -> Path:
    """Fine-tune a backbone for tumour segmentation. Returns best-checkpoint path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pct = f"{int(round(train_fraction * 100))}pct"
    init_tag = "mae" if use_foundation else "scratch"
    tag = f"{arch.name}_{pct}_{init_tag}"
    latest_path = output_dir / f"latest_seg_{tag}.pth"
    best_path = output_dir / f"best_seg_{tag}.pth"

    print(f"Device: {devices.describe(device)}")
    train_loader, test_loader = get_segmentation_dataloaders(
        data_root=data_root,
        train_fraction=train_fraction,
        batch_size=batch_size,
        split_name=split_name,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build the model ONCE, fully initialised, before creating the optimizer.
    model = build_segmentation_model(arch, device, use_foundation, foundation_ckpt)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    lr = arch.lr if lr is None else lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.LR_MIN)
    scaler = devices.make_scaler(device)

    start_epoch, best_metric = 0, -1.0
    if latest_path.exists():
        print(f"==> Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_metric = ckpt["best_metric"]
        print(f"==> Resumed at epoch {start_epoch + 1} (best Dice {best_metric:.4f})")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [LR {current_lr:.2e}]")

        for batch in bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with devices.amp_autocast(device):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            bar.set_postfix({"loss": f"{loss.item():.4f}"})

            del images, labels, logits, loss
            devices.empty_cache(device)

        scheduler.step()
        avg_loss = running / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "arch": arch.name,
                "train_fraction": train_fraction,
                "use_foundation": use_foundation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
            },
            latest_path,
        )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            dice, hd95 = evaluate(
                model, test_loader, device, sw_batch_size=sw_batch_size,
                desc=f"Epoch {epoch + 1}/{epochs} [test]",
            )
            print(f"Validation -- Dice: {dice:.4f} | HD95: {hd95:.4f}")
            if dice > best_metric:
                best_metric = dice
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved: {best_path.name} (Dice {best_metric:.4f})")

    print(f"Fine-tuning complete. Best Dice: {best_metric:.4f} -> {best_path}")
    return best_path
