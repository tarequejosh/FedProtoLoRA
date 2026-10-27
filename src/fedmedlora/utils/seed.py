"""Deterministic random seeding utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
