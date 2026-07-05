"""Utilitários de memória para extração de embeddings em streaming.

Expõe os helpers `free_principal_embs`,
`StreamingEmbDataset`, `streaming_collate`, `make_streaming_loaders`,
`extract_token_embs_chunked`).

O código original assumia que `tokenizer`, `bert_model`, `DEVICE`,
`MAX_SEQ_LEN`, `BATCH_SIZE` e `logger` existiam como globais. Aqui esses valores
são passados como argumentos explícitos.
"""

from __future__ import annotations

import gc
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..config import BATCH_SIZE as _DEFAULT_BS, MAX_LEN as _DEFAULT_MAXLEN


def free_globals(*names: str) -> None:
    """Apaga variáveis globais por nome (equivalente a `free_principal_embs`)."""
    g = globals()
    for n in names:
        g.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class StreamingEmbDataset(Dataset):
    """Dataset que devolve (texto, label). A extração de embedding acontece no
    collate_fn, assim o pico de memória é O(batch × max_len × 768).
    """

    def __init__(self, texts: Iterable[str], labels: Iterable[int]):
        self.texts = list(texts)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_streaming_collate(tokenizer, bert_model, device, max_len: int = _DEFAULT_MAXLEN):
    """Fábrica para o `streaming_collate`."""

    def _collate(batch):
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

    return _collate


def make_streaming_loaders(
    X_tr, y_tr, X_vl, y_vl, X_te, y_te,
    tokenizer, bert_model, device,
    batch_size: int = _DEFAULT_BS, max_len: int = _DEFAULT_MAXLEN,
):
    """Cria três DataLoaders com extração on-the-fly."""
    collate = make_streaming_collate(tokenizer, bert_model, device, max_len=max_len)
    return (
        DataLoader(StreamingEmbDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
                   collate_fn=collate, num_workers=0),
        DataLoader(StreamingEmbDataset(X_vl, y_vl), batch_size=batch_size, shuffle=False,
                   collate_fn=collate, num_workers=0),
        DataLoader(StreamingEmbDataset(X_te, y_te), batch_size=batch_size, shuffle=False,
                   collate_fn=collate, num_workers=0),
    )


def extract_token_embs_chunked(
    texts, tokenizer, bert_model, device,
    max_len: int = _DEFAULT_MAXLEN, batch_size: int = 32, chunk_size: int = 2000,
):
    """Extrai token-level embeddings em chunks."""
    all_embs = []
    bert_model.eval()
    total = len(texts)
    for start in tqdm(range(0, total, chunk_size), desc=f"Chunks de {chunk_size}", leave=False):
        chunk = texts[start : start + chunk_size]
        chunk_embs = []
        for j in range(0, len(chunk), batch_size):
            batch = list(chunk[j : j + batch_size])
            enc = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                h = bert_model(**enc).last_hidden_state
            chunk_embs.append(h.cpu())
        all_embs.append(torch.cat(chunk_embs, 0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(all_embs, 0)


__all__ = [
    "free_globals",
    "StreamingEmbDataset",
    "make_streaming_collate",
    "make_streaming_loaders",
    "extract_token_embs_chunked",
]
