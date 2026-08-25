"""Splits estratificados de treino/validação/teste.

Cobre a célula 23 do notebook (Seção 6.1). Oferece três estratégias:
- `make_random_splits`: split estratificado clássico (70/10/20).
- `make_temporal_splits`: split por tempo (treino antigo, teste recente).
- `make_source_splits`: split out-of-distribution por fonte (GroupShuffleSplit).
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..config import SEED, TEST_SIZE, VAL_SIZE
from ..utils.logging_utils import get_logger

log = get_logger()

# Fração de validação sobre treino+val, derivada do YAML (0.10/0.80 = 0.125).
# round() evita ruído de ponto flutuante que mudaria o ceil() interno do
# train_test_split em DataFrames pequenos (ex.: testes sintéticos).
VAL_SIZE_OF_TRAIN: float = round(VAL_SIZE / (1.0 - TEST_SIZE), 10)


# -- API limpa ----------------------------------------------------------------
def official_split_indices(
    df: pd.DataFrame,
    label_col: str = "label_enc",
    test_size: float = TEST_SIZE,
    val_size_of_train: float = VAL_SIZE_OF_TRAIN,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices (train, val, test) do split oficial 70/10/20 estratificado.

    Fonte única da indexação do split oficial: `make_random_splits` e as
    réplicas externas (sondas de atalho, equalizador) derivam daqui, o que
    mantém índices e split sempre coerentes se os parâmetros mudarem.
    """
    idx = np.arange(len(df))
    y = df[label_col].to_numpy()
    idx_tv, idx_te, y_tv, _ = train_test_split(
        idx, y, test_size=test_size, stratify=y, random_state=seed
    )
    idx_tr, idx_vl, _, _ = train_test_split(
        idx_tv, y_tv, test_size=val_size_of_train, stratify=y_tv, random_state=seed
    )
    return idx_tr, idx_vl, idx_te


def make_random_splits(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label_enc",
    test_size: float = TEST_SIZE,
    val_size_of_train: float = VAL_SIZE_OF_TRAIN,
    seed: int = SEED,
    return_test_df: bool = False,
):
    """Split estratificado train/val/test (default 70/10/20).

    Args:
        df: DataFrame com colunas `text_col` e `label_col`.
        text_col: nome da coluna de texto.
        label_col: nome da coluna de rótulo encoded.
        test_size: fração do teste (sobre o total).
        val_size_of_train: fração de val sobre treino+val (0.125 = 10% do total).
        seed: semente.
        return_test_df: se True, retorna tambem `df_test` com metadados
            (source, category, date_parsed, etc.) preservados na ordem do split.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
        ou (..., df_test) se `return_test_df=True`.
    """
    idx_tr, idx_vl, idx_te = official_split_indices(
        df, label_col=label_col, test_size=test_size,
        val_size_of_train=val_size_of_train, seed=seed,
    )
    y = df[label_col].to_numpy()
    y_tr, y_vl, y_te = y[idx_tr], y[idx_vl], y[idx_te]
    X_tr = df[text_col].iloc[idx_tr].tolist()
    X_vl = df[text_col].iloc[idx_vl].tolist()
    X_te = df[text_col].iloc[idx_te].tolist()
    log.info(f"[Random] Tr={len(X_tr)} Vl={len(X_vl)} Te={len(X_te)}")
    if return_test_df:
        df_test = df.iloc[idx_te].reset_index(drop=True)
        return X_tr, X_vl, X_te, y_tr, y_vl, y_te, df_test
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te


def make_temporal_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Split temporal (train/val/test = 70/10/20 ordenados por data).

    Retorna `None` se <50% de datas válidas.
    """
    if df["date_parsed"].notna().sum() <= len(df) * 0.5:
        log.warning("Temporal split desabilitado (<50% datas válidas).")
        return None

    dfd = (
        df.dropna(subset=["date_parsed"])
        .sort_values("date_parsed")
        .reset_index(drop=True)
    )
    n = len(dfd)
    cut_tr = int(n * train_frac)
    cut_vl = int(n * (train_frac + val_frac))
    df_tr = dfd.iloc[:cut_tr]
    df_vl = dfd.iloc[cut_tr:cut_vl]
    df_te = dfd.iloc[cut_vl:]
    log.info(f"[Temporal] Tr={len(df_tr)} Vl={len(df_vl)} Te={len(df_te)}")
    log.info(
        f'  janelas: Tr até {df_tr["date_parsed"].max().date()}; '
        f'Te a partir de {df_te["date_parsed"].min().date()}'
    )
    return df_tr, df_vl, df_te


def make_source_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    label_col: str = "label_enc",
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Split out-of-distribution por fonte (GroupShuffleSplit).

    Garante que fontes em treino e teste não se sobrepõem. Retorna `None`
    se a coluna `source` está ausente ou com <30% de não-nulos.
    """
    if "source" not in df.columns or df["source"].notna().sum() <= len(df) * 0.3:
        log.info("Source-split desabilitado (coluna ausente ou <30% válidos).")
        return None

    df_src = df.dropna(subset=["source"]).reset_index(drop=True)
    log.info(f"[Source-split] usando {len(df_src)}/{len(df)} amostras")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df_src, df_src[label_col], groups=df_src["source"]))
    df_tr = df_src.iloc[tr_idx].reset_index(drop=True)
    df_te = df_src.iloc[te_idx].reset_index(drop=True)
    overlap = set(df_tr["source"]) & set(df_te["source"])
    log.info(
        f"[Source-split] Tr={len(df_tr)} Te={len(df_te)}  "
        f"fontes sobrepostas: {len(overlap)} (deve ser 0)"
    )
    return df_tr, df_te


__all__ = [
    "make_random_splits",
    "make_temporal_splits",
    "make_source_splits",
]
