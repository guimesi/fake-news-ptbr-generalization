"""Estado dinâmico compartilhado entre etapas do pipeline.

Substitui as variáveis globais que circulavam entre as etapas do pipeline
(`df`, `tokenizer`, `bert_model`, `cls_train`, etc.). Scripts criam um
`ExperimentContext`, passam ele às funções dos subpacotes, e cada função
muta os campos relevantes in-place (ou retorna valores que o script atribui).

Uso típico::

    from fakerecogna2 import ExperimentContext
    from fakerecogna2.data import load_dataset, make_splits
    from fakerecogna2.utils import set_global_seeds

    ctx = ExperimentContext()
    set_global_seeds(ctx.seed)
    ctx.df = load_dataset()
    make_splits(ctx)            # popula ctx.df_train, ctx.y_test, etc.

Cada subpacote documenta quais campos ele consome e quais ele produz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import SEED, MAX_LEN


@dataclass
class ExperimentContext:
    """Container mutável pro estado do experimento."""

    # --- Hardware / determinismo --------------------------------------------
    seed: int = SEED
    device: str = field(default_factory=lambda: _default_device())
    max_seq_len: int = MAX_LEN

    # --- Dados brutos e splits ----------------------------------------------
    df: pd.DataFrame | None = None
    df_train: pd.DataFrame | None = None
    df_val: pd.DataFrame | None = None
    df_test: pd.DataFrame | None = None
    X_train_text: list[str] | None = None
    X_val_text: list[str] | None = None
    X_test_text: list[str] | None = None
    y_train: np.ndarray | None = None
    y_val: np.ndarray | None = None
    y_test: np.ndarray | None = None

    # --- Vetorizações -----------------------------------------------------
    tfidf_vectorizer: Any | None = None
    X_train_tfidf: Any | None = None
    X_val_tfidf: Any | None = None
    X_test_tfidf: Any | None = None

    # --- Tokenizer / backbones ---------------------------------------------
    tokenizer: Any | None = None
    bert_model: Any | None = None   # backbone para extração de embeddings
    bert_clf: Any | None = None     # BERTimbau fine-tuned para classificação

    # --- Embeddings pré-computados ----------------------------------------
    cls_train: np.ndarray | None = None
    cls_val: np.ndarray | None = None
    cls_test: np.ndarray | None = None
    token_train: np.ndarray | None = None
    token_val: np.ndarray | None = None
    token_test: np.ndarray | None = None

    # --- Modelos treinados (clas. clássicos, CNN, LSTM, etc.) -------------
    models: dict[str, Any] = field(default_factory=dict)

    # --- Predições e probabilidades, indexadas por nome de modelo --------
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    probabilities: dict[str, np.ndarray] = field(default_factory=dict)

    # --- Métricas agregadas (replica `RESULTS`) ---------------
    results: dict[str, Any] = field(default_factory=dict)

    # --- Slot genérico p/ qualquer estado extra ---------------------------
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------------
    # Ponte para namespaces globais (útil pra integrações ad-hoc)
    # ------------------------------------------------------------------------
    def to_namespace(self) -> dict[str, Any]:
        """Exporta campos para um dict com os nomes globais.

        Útil pra rodar código legado que usa nomes bare como ``df``,
        ``tokenizer``, ``DEVICE``, ``MAX_SEQ_LEN``. Campos ``None`` não
        são exportados.
        """
        ns: dict[str, Any] = {}
        mapping = {
            "df": "df",
            "df_train": "df_train",
            "df_val": "df_val",
            "df_test": "df_test",
            "X_train_text": "X_train_text",
            "X_val_text": "X_val_text",
            "X_test_text": "X_test_text",
            "y_train": "y_train",
            "y_val": "y_val",
            "y_test": "y_test",
            "tfidf_vectorizer": "tfidf_vectorizer",
            "X_train_tfidf": "X_train_tfidf",
            "X_val_tfidf": "X_val_tfidf",
            "X_test_tfidf": "X_test_tfidf",
            "tokenizer": "tokenizer",
            "bert_model": "bert_model",
            "bert_clf": "bert_clf",
            "cls_train": "cls_train",
            "cls_val": "cls_val",
            "cls_test": "cls_test",
            "token_train": "token_train",
            "token_val": "token_val",
            "token_test": "token_test",
            "device": "DEVICE",
            "max_seq_len": "MAX_SEQ_LEN",
            "seed": "SEED",
        }
        for attr, var in mapping.items():
            value = getattr(self, attr)
            if value is not None:
                ns[var] = value
        return ns

    def update_from_namespace(self, ns: dict[str, Any]) -> None:
        """Atualiza o contexto a partir de um dict de nomes globais.

        Olha o ``ns`` e copia de volta qualquer variável cujo nome bate
        com um campo conhecido. Útil pra ler resultados de scripts ad-hoc
        que produziram variáveis com nomes legado.
        """
        reverse_mapping = {
            "df": "df",
            "df_train": "df_train",
            "df_val": "df_val",
            "df_test": "df_test",
            "X_train_text": "X_train_text",
            "X_val_text": "X_val_text",
            "X_test_text": "X_test_text",
            "y_train": "y_train",
            "y_val": "y_val",
            "y_test": "y_test",
            "tfidf_vectorizer": "tfidf_vectorizer",
            "X_train_tfidf": "X_train_tfidf",
            "X_val_tfidf": "X_val_tfidf",
            "X_test_tfidf": "X_test_tfidf",
            "tokenizer": "tokenizer",
            "bert_model": "bert_model",
            "bert_clf": "bert_clf",
            "cls_train": "cls_train",
            "cls_val": "cls_val",
            "cls_test": "cls_test",
            "token_train": "token_train",
            "token_val": "token_val",
            "token_test": "token_test",
        }
        for var, attr in reverse_mapping.items():
            if var in ns and ns[var] is not None:
                setattr(self, attr, ns[var])


def _default_device() -> str:
    """Tenta importar torch sem falhar caso ele não esteja instalado."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


__all__ = ["ExperimentContext"]
