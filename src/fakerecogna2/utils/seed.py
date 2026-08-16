"""Reprodutibilidade: fixa todas as seeds (numpy, torch, cudnn).

Equivalente ao bloco final da célula 1.2 e à função `set_seed` da célula 1.3.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from ..config import SEED


def set_seed(s: int) -> None:
    """Fixa seeds de numpy, random e torch para uma execução individual.

    Cópia fiel da função `set_seed(s)` da célula 1.3 do notebook.
    """
    random.seed(s)
    np.random.seed(s)
    if _HAS_TORCH:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)


def set_global_seeds(s: int = SEED, deterministic_cudnn: bool = True) -> None:
    """Setup determinístico global, equivalente à célula 1.2.

    Inclui PYTHONHASHSEED, cudnn.deterministic e cudnn.benchmark=False.
    """
    os.environ["PYTHONHASHSEED"] = str(s)
    set_seed(s)
    if _HAS_TORCH and torch.cuda.is_available() and deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed", "set_global_seeds"]
