"""Ensembles: grid search de pesos + ensemble no dataset Extrativo."""

from __future__ import annotations

from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ..config import SEED
from ..utils.io_utils import RESULTS
from ..utils.logging_utils import get_logger
from .training import evaluate_model, train_model

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def get_val_probs(
    model: nn.Module, vl_loader: DataLoader, device: str = "cpu"
) -> torch.Tensor:
    """Forward passes no val_loader, retorna [N, num_classes] de softmax."""
    model.eval()
    ps = []
    with torch.no_grad():
        for bx, _ in vl_loader:
            ps.append(torch.softmax(model(bx.to(device)), 1).cpu())
    return torch.cat(ps, 0)


def grid_search_2model(
    vp_a: torch.Tensor,
    vp_b: torch.Tensor,
    y_val: np.ndarray,
    step: float = 0.05,
) -> tuple[float, float]:
    """Procura α∈[0,1] que maximiza F1 macro de α·a + (1-α)·b."""
    best_a, best_f = 0.5, 0.0
    for a in np.arange(0.0, 1.0 + step / 2, step):
        p = a * vp_a + (1 - a) * vp_b
        f = f1_score(y_val, p.argmax(1).numpy(), average="macro", zero_division=0)
        if f > best_f:
            best_a, best_f = float(a), float(f)
    return best_a, best_f


def grid_search_3model(
    vp_a: torch.Tensor,
    vp_b: torch.Tensor,
    vp_c: torch.Tensor,
    y_val: np.ndarray,
    step: float = 0.1,
) -> tuple[tuple[float, float, float], float]:
    """Grid (a,b,c) com c=max(0,1-a-b) que maximiza F1 macro."""
    best_abc, best_f = (1 / 3, 1 / 3, 1 / 3), 0.0
    for a in np.arange(0.0, 1.0 + step / 2, step):
        for b in np.arange(0.0, 1.0 - a + step / 2, step):
            c = max(0.0, 1 - a - b)
            p = a * vp_a + b * vp_b + c * vp_c
            f = f1_score(y_val, p.argmax(1).numpy(), average="macro", zero_division=0)
            if f > best_f:
                best_abc, best_f = (float(a), float(b), float(c)), float(f)
    return best_abc, best_f


def register_ensemble(
    name: str, probs: torch.Tensor, y_test: np.ndarray, inference_ms: float = 0.0
) -> np.ndarray:
    """Calcula métricas das probabilidades e registra em RESULTS."""
    preds = probs.argmax(1).numpy()
    RESULTS[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "F1": f1_score(y_test, preds, average="macro", zero_division=0),
        "Inference (ms)": inference_ms,
    }
    return preds


def train_ensemble_on_variant(
    cnn_cls: Type[nn.Module],
    lstm_cls: Type[nn.Module],
    cnn_kwargs: dict[str, Any],
    lstm_kwargs: dict[str, Any],
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    te_loader: DataLoader,
    y_test: np.ndarray,
    device: str = "cpu",
    epochs: int = 15,
    suffix: str = "ext",
    save_as: str | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Treina CNN+LSTM em loaders alternativos (ex.: extrativa, equalizada).

    Returns:
        dict com Accuracy, F1 (float) e y_pred (np.ndarray) do ensemble.
    """
    cnn = cnn_cls(**cnn_kwargs)
    cnn, _ = train_model(
        cnn, tr_loader, vl_loader, device=device, epochs=epochs,
        model_name=f"CNN_{suffix}", seed=seed,
    )
    _, _, _, cnn_p = evaluate_model(cnn, te_loader, device=device, name=f"CNN_{suffix}")

    lstm = lstm_cls(**lstm_kwargs)
    lstm, _ = train_model(
        lstm, tr_loader, vl_loader, device=device, epochs=epochs,
        model_name=f"LSTM_{suffix}", seed=seed,
    )
    _, _, _, lstm_p = evaluate_model(lstm, te_loader, device=device, name=f"LSTM_{suffix}")

    ens_preds = ((cnn_p + lstm_p) / 2).argmax(1).numpy()
    acc = accuracy_score(y_test, ens_preds)
    f1 = f1_score(y_test, ens_preds, average="macro", zero_division=0)

    # Liberar GPU
    del cnn, lstm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"Accuracy": float(acc), "F1": float(f1), "y_pred": ens_preds}


__all__ = [
    "get_val_probs",
    "grid_search_2model",
    "grid_search_3model",
    "register_ensemble",
    "train_ensemble_on_variant",
]
