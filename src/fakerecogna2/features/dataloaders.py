"""DataLoaders PyTorch a partir dos embeddings."""

from __future__ import annotations


from ..config import BATCH_SIZE




# -- API limpa ----------------------------------------------------------------
def make_loaders(
    emb_train, emb_val, emb_test, y_train, y_val, y_test, batch_size: int = BATCH_SIZE
):
    """3 DataLoaders (train/val/test) a partir de embeddings tensorizados."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _to_t(y):
        return torch.tensor(np.asarray(y), dtype=torch.long)

    return (
        DataLoader(TensorDataset(emb_train, _to_t(y_train)), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(emb_val, _to_t(y_val)), batch_size=batch_size),
        DataLoader(TensorDataset(emb_test, _to_t(y_test)), batch_size=batch_size),
    )


__all__ = [
    "make_loaders",
]
