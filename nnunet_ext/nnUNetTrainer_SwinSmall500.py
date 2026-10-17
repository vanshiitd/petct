"""SwinUNETR-small inside nnU-Net, initialised from the MAE foundation weights.

Phase VI asks whether the repair agent transfers to a backbone it was not built
on. That needs a second backbone trained under conditions otherwise identical to
the nnU-Net baseline: same Dataset505, same preprocessing, same
`nnUNetPlans_MAE` plans, same fold-0 738/76 split, same 500 epochs. Only the
network differs.

The network is built by `petct.models.build_swin_unetr` with `ARCHS["small"]`,
so there is exactly one definition of the architecture in the project and this
trainer cannot drift from the standalone code. Weights come from
`weights/swin_small_mae_best.pth` through `petct.models.load_foundation_weights`,
which fails loudly rather than silently transferring nothing on a prefix
mismatch.

Two things nnU-Net does by default that must be switched off or checked:

  deep supervision  SwinUNETR returns a single tensor, not a resolution pyramid,
                    so deep supervision has to be disabled or the loss wrapper
                    will try to unpack it. `_build_loss` is left alone; the flag
                    is what nnU-Net reads.

  patch size        Swin partitions the volume into windows and needs the patch
                    divisible by 32 on every axis (2 merging stages of 2, times
                    a window of 8... in practice 32). [192, 96, 128] satisfies
                    this. It is asserted rather than assumed, because a plans
                    change would otherwise surface as an obscure reshape error
                    deep inside the attention block.

Optimiser: nnU-Net's SGD at 1e-2 is kept by default, since the point is to
change one thing at a time. `PETCT_SWIN_ADAMW=1` switches to AdamW at 1e-4,
which is what `petct/config.py` uses for this architecture standalone, for use
if the SGD pilot is unstable.

    $env:nnUNet_extTrainer = 'D:\\data\\petct\\nnunet_ext'
    nnUNetv2_train 505 3d_fullres 0 -p nnUNetPlans_MAE -tr nnUNetTrainer_SwinSmall500
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_500epochs,
)

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

WEIGHTS = _REPO / "weights" / "swin_small_mae_best.pth"


class nnUNetTrainer_SwinSmall500(nnUNetTrainer_500epochs):
    """500 epochs, SwinUNETR-small backbone, MAE-initialised, no deep supervision."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # SwinUNETR emits one tensor, not a pyramid.
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        from petct.config import ARCHS, IN_CHANNELS
        from petct.models import build_swin_unetr, load_foundation_weights

        patch = tuple(int(x) for x in configuration_manager.patch_size)
        bad = [x for x in patch if x % 32]
        if bad:
            raise SystemExit(
                f"patch size {patch} is not divisible by 32 on every axis "
                f"({bad}); SwinUNETR window partitioning would fail. Re-plan or "
                f"pad before training."
            )
        if num_input_channels != IN_CHANNELS:
            raise SystemExit(
                f"plans give {num_input_channels} input channels but "
                f"petct.config.IN_CHANNELS is {IN_CHANNELS}; build_swin_unetr "
                f"would construct a stem of the wrong width."
            )

        arch = ARCHS["small"]
        model = build_swin_unetr(arch, out_channels=num_output_channels)
        # Deviation, recorded in final_eval_plan.md section 7: build_swin_unetr
        # sets use_checkpoint=True, which recomputes activations in the backward
        # pass to save memory. Gradient checkpointing is an exact recomputation,
        # so gradients and the trained model are unchanged -- it buys memory with
        # compute. At batch 2 and patch 192x96x128 it costs 0.727 s/step against
        # 0.599 s/step, which is 25.2 h against 20.8 h for 500 epochs. Peak
        # memory rises 11.6 -> 17.0 GB, inside a 32 GB card. Turned off so the
        # run fits the agreed wall-clock budget.
        for m in model.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = False
        if WEIGHTS.exists():
            model = load_foundation_weights(model, WEIGHTS, arch, verbose=True)
        else:
            raise SystemExit(f"foundation checkpoint not found: {WEIGHTS}")
        return model

    def configure_optimizers(self):
        if os.environ.get("PETCT_SWIN_ADAMW") != "1":
            return super().configure_optimizers()
        # Deviation switch, used only if the SGD pilot is unstable. Reported in
        # the Phase VI report when it is used.
        from torch.optim.lr_scheduler import CosineAnnealingLR
        opt = torch.optim.AdamW(self.network.parameters(), lr=1e-4,
                                weight_decay=1e-5)
        sched = CosineAnnealingLR(opt, T_max=self.num_epochs)
        self.print_to_log_file("optimiser: AdamW 1e-4, cosine (PETCT_SWIN_ADAMW=1)")
        return opt, sched

    def _get_deep_supervision_scales(self):
        return None

    def set_deep_supervision_enabled(self, enabled: bool):
        """No-op: SwinUNETR has no `.decoder` to toggle.

        nnU-Net's implementation reaches for `self.network.decoder
        .deep_supervision`, which is specific to its own architecture. On
        SwinUNETR that raises `AttributeError: 'SwinUNETR' object has no
        attribute 'decoder'` inside the dataloader worker, which surfaces only
        as "One or more background workers are no longer alive" -- the real
        message is several frames up. Deep supervision is off for this trainer
        in any case, so there is nothing to toggle.
        """
        return
