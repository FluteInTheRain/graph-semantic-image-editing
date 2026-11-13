"""The single seeding function of the project (ADR-0011)."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (and CUDA if present) so a run can be repeated.

    torch is imported here, not at the top, so modules that only need numpy can
    import this file without installing torch.
    """
    import torch

    random.seed(seed)
    # Our own code passes an explicit np.random.Generator; this only covers library
    # code that still draws from NumPy's global generator.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
