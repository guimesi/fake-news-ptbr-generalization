"""5-fold CV com split interno de validação, sem leakage (cell 44)."""

from __future__ import annotations

from typing import Any, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..config import BATCH_SIZE, CV_FOLDS, SEED
from ..models.training import train_model
from ..utils.io_utils import cleanup, save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def cross_validate_ensemble(
    all_embeddings: torch.Tensor,
    all_y: np.ndarray,
    all_texts: list[str],
    model_classes: list[tuple[Type[nn.Module], dict[str, Any], str]],
    n_splits: int = CV_FOLDS,
    val_frac: float = 0.10,
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
    seed: int = SEED,
    save_as: str | None = "13_cross_validation",
) -> pd.DataFrame:
    """K-fold CV com split interno de val (sem leakage).

    Args:
        all_embeddings: tensor [N, max_len, hidden_size] concatenado dos 3 splits.
        all_y: labels concatenados.
        all_texts: textos correspondentes (usado p/ split estratificado).
        model_classes: lista de (classe, kwargs, nome). Faz ensemble por média de probs.
        val_frac: fração do trainval pra validation interna.

    Returns:
        DataFrame com fold, acc, f1_macro + linhas 'mean' e 'std'.
    """
    full_ds = TensorDataset(all_embeddings, torch.tensor(all_y, dtype=torch.long))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(all_texts, all_y)):
        rng = np.random.RandomState(seed + fold)
        rng.shuffle(trainval_idx)
        n_val = int(len(trainval_idx) * val_frac)
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]

        ld_tr = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True)
        ld_vl = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size)
        ld_te = DataLoader(Subset(full_ds, test_idx), batch_size=batch_size)

        trained = []
        for cls, kwargs, name in model_classes:
            m = cls(**kwargs)
            m, _ = train_model(
                m, ld_tr, ld_vl, device=device, epochs=epochs,
                model_name=f"CV_{name}_f{fold+1}", seed=seed + fold,
            )
            m.eval()
            trained.append(m)

        ps, ls = [], []
        with torch.no_grad():
            for bx, by in ld_te:
                bx = bx.to(device)
                probs = sum(torch.softmax(m(bx), 1) for m in trained) / len(trained)
                ps.extend(probs.argmax(1).cpu().numpy())
                ls.extend(by.numpy())
        rows.append(
            {
                "fold": fold + 1,
                "acc": accuracy_score(ls, ps),
                "f1_macro": f1_score(ls, ps, average="macro", zero_division=0),
            }
        )
        log.info(
            f'Fold {fold+1}: Acc={rows[-1]["acc"]:.4f} '
            f'F1={rows[-1]["f1_macro"]:.4f}'
        )

        for m in trained:
            del m
        cleanup()

    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"fold": "mean", **df[["acc", "f1_macro"]].mean().to_dict()},
            {"fold": "std", **df[["acc", "f1_macro"]].std().to_dict()},
        ]
    )
    df = pd.concat([df, summary], ignore_index=True)
    print(tabulate(df.round(4), headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


__all__ = [
    "cross_validate_ensemble",
]
