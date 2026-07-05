"""Split anti-vies: estratifica por (categoria, periodo, classe).

A ideia e que cada split (train/val/test) tenha proporcao similar de cada
combinacao (categoria_top, ano_bucket, label). Isso reduz a chance de
shortcuts onde uma categoria/ano correlaciona quase perfeitamente com a
classe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import SEED
from ..utils.logging_utils import get_logger

log = get_logger()


def _make_strata(
    df: pd.DataFrame,
    label_col: str,
    category_col: str = "category",
    date_col: str = "date_parsed",
    top_k_categories: int = 10,
) -> pd.Series:
    """Constroi label de estratificacao = label x cat_bucket x year_bucket."""
    # Categoria: top-K mais frequentes; resto vira "other"
    if category_col in df.columns:
        top_cats = (
            df[category_col]
            .fillna("missing")
            .value_counts()
            .head(top_k_categories)
            .index.tolist()
        )
        cat_bucket = df[category_col].fillna("missing").where(
            df[category_col].isin(top_cats), other="other"
        )
    else:
        cat_bucket = pd.Series(["all"] * len(df), index=df.index)

    # Periodo: quartis de date_parsed.year. Anos podem concentrar (ex.: 2020
    # pandemia); usa qcut(duplicates='drop') para que o numero de buckets se
    # adapte dinamicamente. Datas ausentes vao para bucket 'missing'.
    if date_col in df.columns:
        years = pd.to_datetime(df[date_col], errors="coerce").dt.year
        valid = years.notna()
        if valid.sum() > 100:
            try:
                cat = pd.qcut(
                    years[valid], q=4, duplicates="drop",
                )
                # Labels = P1..PN onde N = numero real de bins gerados
                n_bins = cat.cat.categories.size
                cat = cat.cat.rename_categories(
                    [f"P{i+1}" for i in range(n_bins)]
                )
                year_bucket = pd.Series("missing", index=df.index, dtype=object)
                year_bucket.loc[valid] = cat.astype(str)
                if n_bins < 4:
                    log.info(
                        f"[anti-bias] anos concentrados -> {n_bins} buckets "
                        f"temporais (em vez de 4)"
                    )
            except ValueError:
                year_bucket = pd.Series(["all"] * len(df), index=df.index)
        else:
            year_bucket = pd.Series(["all"] * len(df), index=df.index)
    else:
        year_bucket = pd.Series(["all"] * len(df), index=df.index)

    strata = (
        df[label_col].astype(str) + "|" + cat_bucket.astype(str) + "|"
        + year_bucket.astype(str)
    )
    return strata


def make_anti_bias_splits(
    df: pd.DataFrame,
    text_col: str = "text_proc",
    label_col: str = "label_enc",
    category_col: str = "category",
    date_col: str = "date_parsed",
    test_size: float = 0.20,
    val_size_of_train: float = 0.125,
    seed: int = SEED,
    top_k_categories: int = 10,
    min_stratum_size: int = 5,
):
    """Split estratificado simultaneamente por (label, categoria, periodo).

    Estratos com <`min_stratum_size` exemplos sao fundidos no estrato 'rare'.

    Returns:
        (X_tr, X_vl, X_te, y_tr, y_vl, y_te, df_test_meta) - df_test_meta
        contem source/category/date_parsed para auditoria.
    """
    text_to_use = text_col if text_col in df.columns else "text"
    strata = _make_strata(
        df, label_col=label_col, category_col=category_col,
        date_col=date_col, top_k_categories=top_k_categories,
    )

    # Funde estratos pequenos
    counts = strata.value_counts()
    rare = counts[counts < min_stratum_size].index.tolist()
    if rare:
        strata = strata.where(~strata.isin(rare), other="rare_stratum")
        log.info(f"[anti-bias] {len(rare)} estratos pequenos fundidos em 'rare_stratum'")

    log.info(f"[anti-bias] {strata.nunique()} estratos efetivos (label x cat x periodo)")

    idx = np.arange(len(df))
    idx_tv, idx_te = train_test_split(
        idx, test_size=test_size, stratify=strata.values, random_state=seed,
    )
    strata_tv = strata.iloc[idx_tv].values
    idx_tr, idx_vl = train_test_split(
        idx_tv, test_size=val_size_of_train,
        stratify=strata_tv, random_state=seed,
    )
    X_tr = df[text_to_use].iloc[idx_tr].tolist()
    X_vl = df[text_to_use].iloc[idx_vl].tolist()
    X_te = df[text_to_use].iloc[idx_te].tolist()
    y_tr = df[label_col].iloc[idx_tr].to_numpy()
    y_vl = df[label_col].iloc[idx_vl].to_numpy()
    y_te = df[label_col].iloc[idx_te].to_numpy()
    df_te = df.iloc[idx_te].reset_index(drop=True)
    log.info(f"[anti-bias] Tr={len(X_tr)} Vl={len(X_vl)} Te={len(X_te)}")
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te, df_te


__all__ = ["make_anti_bias_splits"]
