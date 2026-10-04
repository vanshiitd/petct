"""A 500-epoch trainer that is `nnUNetTrainer_500epochs` under a different name.

The from-scratch control has to be identical to the MAE-initialised run in every
respect except the initialisation, which is set by *omitting* `-pretrained_weights`
rather than by anything in the trainer. So the two runs would share a trainer name,
and nnU-Net derives its output folder from

    nnUNet_results/<dataset>/<trainer>__<plans>__<configuration>/fold_N

which means the control would write its checkpoints, logs and validation
predictions straight over the pretrained run's. Renaming the trainer is what keeps
them apart.

Subclassing rather than copying keeps the two runs provably identical: there is no
second copy of the training schedule to drift. The only difference between
`nnUNetTrainer_500epochs` and this class is `__name__`.

nnU-Net finds it through the `nnUNet_extTrainer` environment variable, so it stays
in this repository instead of being dropped into site-packages:

    $env:nnUNet_extTrainer = 'D:\\data\\petct\\nnunet_ext'
    nnUNetv2_train 505 3d_fullres 0 -p nnUNetPlans_MAE -tr nnUNetTrainer_500epochs_scratch
"""
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_500epochs,
)


class nnUNetTrainer_500epochs_scratch(nnUNetTrainer_500epochs):
    """`nnUNetTrainer_500epochs` with a distinct name, so results land elsewhere."""
