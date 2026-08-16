"""Orquestrador do subpacote `preprocessing`.

Aplica pré-processamento (`text_proc`) e em seguida faz os splits sobre
o texto preprocessado. Opcionalmente gera versões equalizadas (sumarização,
metatexto removido).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._context import ExperimentContext
from ..data.splits import (
    make_random_splits,
    make_source_splits,
    make_temporal_splits,
)
from .equalizer import build_nometatext_dataset
from .paraphrasing import build_equalized_dataset
from .text_cleaning import apply_preprocessing


def run(
    ctx: ExperimentContext,
    method: str = "base",
    do_equalized: bool = False,
    do_nometatext: bool = False,
    do_splits: bool = True,
    text_col: str = "text",
    out_col: str = "text_proc",
) -> ExperimentContext:
    """Pré-processa o df principal e (opcionalmente) gera splits + variantes.

    Args:
        ctx: contexto com `ctx.df` populado (rode `data.run_pipeline` antes).
        method: 'base' | 'stem' | 'lemma'.
        do_equalized: gera versão sumarizada em `ctx.extras['df_equalized']`.
        do_nometatext: gera versão sem metatexto em `ctx.extras['df_nometatext']`.
        do_splits: faz random / temporal / source splits sobre `text_proc`
            e popula `ctx.X_train_text/X_val_text/X_test_text` e
            `ctx.y_train/y_val/y_test`.
        text_col/out_col: nomes das colunas de entrada/saída.
    """
    if ctx.df is None:
        raise RuntimeError("ctx.df está vazio — rode `data.run_pipeline(ctx)` antes.")

    ctx.df = apply_preprocessing(ctx.df, text_col=text_col, out_col=out_col, method=method)
    if "df_extr" in ctx.extras and isinstance(ctx.extras["df_extr"], pd.DataFrame):
        ctx.extras["df_extr"] = apply_preprocessing(
            ctx.extras["df_extr"], text_col=text_col, out_col=out_col, method=method
        )

    if do_splits:
        X_tr, X_vl, X_te, y_tr, y_vl, y_te, df_test_meta = make_random_splits(
            ctx.df, text_col=out_col, return_test_df=True,
        )
        ctx.X_train_text, ctx.X_val_text, ctx.X_test_text = X_tr, X_vl, X_te
        ctx.y_train = np.asarray(y_tr)
        ctx.y_val = np.asarray(y_vl)
        ctx.y_test = np.asarray(y_te)
        # Metadata do teste (source, category, date_parsed) para analise
        # de erros estratificada (Cap. 5.9).
        ctx.extras["df_test_meta"] = df_test_meta

        temporal = make_temporal_splits(ctx.df)
        if temporal is not None:
            ctx.extras["splits_temporal"] = temporal
        source = make_source_splits(ctx.df)
        if source is not None:
            ctx.extras["splits_source"] = source

        if "df_extr" in ctx.extras and isinstance(ctx.extras["df_extr"], pd.DataFrame):
            Xe_tr, Xe_vl, Xe_te, ye_tr, ye_vl, ye_te = make_random_splits(
                ctx.extras["df_extr"], text_col=out_col
            )
            ctx.extras["splits_extrativa"] = {
                "X_train": Xe_tr,
                "X_val": Xe_vl,
                "X_test": Xe_te,
                "y_train": np.asarray(ye_tr),
                "y_val": np.asarray(ye_vl),
                "y_test": np.asarray(ye_te),
            }

    if do_equalized:
        ctx.extras["df_equalized"] = build_equalized_dataset(ctx.df)
    if do_nometatext:
        ctx.extras["df_nometatext"] = build_nometatext_dataset(ctx.df)

    return ctx


__all__ = ["run"]
