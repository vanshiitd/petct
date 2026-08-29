"""PET-CT tumour segmentation: MAE pretraining and supervised fine-tuning.

Converted from the original notebook collection. The full logic lives here once
and is driven by scripts/ entrypoints; see CONVERSION_NOTES.md for the mapping
from each old notebook to its replacement command.

Light imports only at package level -- torch/MONAI are pulled in lazily by the
submodules that need them, so preprocessing scripts stay runnable on a machine
without a deep-learning stack installed.
"""

__version__ = "1.0.0"

from . import config  # noqa: F401  (safe: stdlib only)

__all__ = ["config"]
