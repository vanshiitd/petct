"""Stage 1: self-supervised MAE pretraining on unlabelled PET/CT volumes."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from . import config, devices
from .config import ArchSpec
from .data import get_pretraining_dataloader
from .models import build_pretrain_model


def run_pretraining(
    arch: ArchSpec,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = config.PRETRAIN_EPOCHS,
    batch_size: int = 16,
    lr: float | None = None,
    num_workers: int = 8,
    data_parallel: bool = True,
) -> Path:
    """Train a backbone to reconstruct masked PET/CT volumes.

    Returns the path of the best checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / f"{arch.name}_mae_latest.pth"
    best_path = output_dir / f"{arch.name}_mae_best.pth"

    print(f"Device: {devices.describe(device)}")
    model = build_pretrain_model(arch)

    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    lr = arch.lr if lr is None else lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.LR_MIN)
    scaler = devices.make_scaler(device)

    loader = get_pretraining_dataloader(
        data_root, batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    start_epoch, best_loss = 0, float("inf")
    if latest_path.exists():
        print(f"=> Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["best_loss"]
        print(f"=> Resumed at epoch {start_epoch + 1} (best loss {best_loss:.4f})")
    else:
        print(f"=> Starting {arch.name} pretraining from scratch")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        bar = tqdm(loader, total=len(loader),
                   desc=f"Epoch {epoch + 1}/{epochs} [LR {current_lr:.2e}]")

        for batch in bar:
            inputs = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with devices.amp_autocast(device):
                loss, _ = model(inputs)
                loss = loss.mean()  # DataParallel returns one loss per device

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Cross-modal masking can produce very large gradients early on.
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            bar.set_postfix({"loss": f"{loss.item():.4f}"})

            del inputs, loss
            devices.empty_cache(device)

        scheduler.step()
        avg_loss = running / max(1, len(loader))
        print(f"[Epoch {epoch + 1}] average loss: {avg_loss:.4f}")

        state = {
            "epoch": epoch + 1,
            "arch": arch.name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(state, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            state["best_loss"] = best_loss
            torch.save(state, best_path)
            print(f"*** New best: {best_loss:.4f} -> {best_path.name} ***")

    print(f"Pretraining complete. Best checkpoint: {best_path}")
    return best_path
