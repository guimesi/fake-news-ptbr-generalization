"""Extração de embeddings sob demanda (streaming): memória constante.

Usado em ablações e cross-dataset onde manter todos os embeddings em RAM
ao mesmo tempo estouraria ~20 GB. Custo: ~10-15% mais lento por época vs
pré-compute, mas memória O(batch × max_len × dim).
"""

from __future__ import annotations

import gc
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import BATCH_SIZE, MAX_LEN
from ..utils.logging_utils import get_logger

log = get_logger()


class StreamingEmbDataset(Dataset):
    """Dataset que devolve (texto, label), embedding é extraído no collate.

    Memória: O(batch × max_len × hidden_size), constante. Velocidade:
    ~15% mais lento por época que pré-computar em RAM.
    """

    def __init__(self, texts, labels, max_len: int = MAX_LEN):
        self.texts = list(texts)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_streaming_collate(
    tokenizer, bert_model, device: str = "cpu", max_len: int = MAX_LEN
) -> Callable:
    """Cria um collate_fn que tokeniza o batch e extrai embeddings.

    Args:
        tokenizer: tokenizer HuggingFace.
        bert_model: backbone (já em `device` e `.eval()`).
        device: 'cuda' ou 'cpu'.
        max_len: comprimento máximo de sequência.

    Returns:
        Função pronta pra usar como `collate_fn` num `DataLoader`.
    """
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            embs = bert_model(**enc).last_hidden_state
        return embs.cpu(), torch.stack(labels)

    return collate


def make_streaming_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    tokenizer,
    bert_model,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """3 DataLoaders (train/val/test) com extração on-the-fly de embeddings."""
    collate = make_streaming_collate(tokenizer, bert_model, device=device, max_len=max_len)
    ds_tr = StreamingEmbDataset(X_train, y_train, max_len=max_len)
    ds_vl = StreamingEmbDataset(X_val, y_val, max_len=max_len)
    ds_te = StreamingEmbDataset(X_test, y_test, max_len=max_len)
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0),
        DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
        DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
    )


def free_embeddings(*tensor_names: str, namespace: dict | None = None) -> None:
    """Libera embeddings da RAM. Use antes de começar uma ablação que reextrai.

    Em uso simples (com `ExperimentContext`), prefira fazer
    ``ctx.token_train = None`` diretamente. Esta função é pra namespace
    global (passe o `globals()` do caller via `namespace`).
    """
    if namespace is not None:
        for name in tensor_names:
            namespace.pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Embeddings liberados da RAM.")


__all__ = [
    "StreamingEmbDataset",
    "make_streaming_collate",
    "make_streaming_loaders",
    "free_embeddings",
]
