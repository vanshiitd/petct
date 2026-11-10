"""Phase XX: fine-tune the FDG nnU-Net on PSMA from a full FDG checkpoint.

`-pretrained_weights` skips the segmentation head (`.seg_layers.`), which is
the right default across tasks but not here: FDG and PSMA share the same
two-class tumour task, so the head is transferred too. The trainer therefore
loads the whole FDG state dict itself, strictly, before training starts.

The source checkpoint is taken from `PSMAFT_INIT`, a path template with `{fold}`
filled by FDG fold 3 + this fold (`reports/phase20_segmenter_selfimprove_plan.md`):

    $env:PSMAFT_INIT = 'D:\\data\\nnunet\\results\\Dataset505_AutoPET_MAEprep_v2\\
        nnUNetTrainer_500epochs__nnUNetPlans_MAE__3d_fullres\\fold_{fold}\\checkpoint_final.pth'
    nnUNetv2_train 507 3d_fullres 0 -p nnUNetPlans_MAE -tr nnUNetTrainer_PSMAft
"""
import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

FDG_FOLD_OFFSET = 3


class nnUNetTrainer_PSMAft(nnUNetTrainer):
    """nnU-Net defaults, except 100 epochs, initial LR 3e-3 and a full FDG init."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.initial_lr = 3e-3

    def initialize(self):
        super().initialize()
        src = os.environ["PSMAFT_INIT"].format(fold=FDG_FOLD_OFFSET + int(self.fold))
        sd = torch.load(src, map_location=self.device, weights_only=False)["network_weights"]
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, "_orig_mod", net)
        net.load_state_dict(sd, strict=True)
        self.print_to_log_file(f"Phase XX: loaded all {len(sd)} tensors from {src}")
