"""Helpers compartilhados pelos scripts numerados.

Centraliza o setup (`ExperimentContext`, dirs, seeds, logger) e a pré-fase
do pipeline (data → preprocessing → features) que praticamente todo script
precisa antes da sua etapa específica.

Uso típico em um script::

    from _pipeline import prepare_through_features

    ctx = prepare_through_features()              # data + preproc + embeddings + TF-IDF
    # ...etapa específica usando ctx...

Para etapas que NÃO precisam de embeddings BERT (ex.: log-odds só),
use a versão mais leve::

    from _pipeline import prepare_through_preprocessing

    ctx = prepare_through_preprocessing()
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fakerecogna2 import ExperimentContext
from fakerecogna2.config import BATCH_SIZE, ensure_dirs
from fakerecogna2.data import run_pipeline as data_pipeline
from fakerecogna2.features import (
    EmbeddingExtractor,
    extract_all_embeddings,
    fit_tfidf,
    make_loaders,
)
from fakerecogna2.preprocessing import run_pipeline as preprocessing_pipeline
from fakerecogna2.utils import get_logger, set_global_seeds


def setup_context(seed: int | None = None) -> ExperimentContext:
    """Cria contexto + dirs + seeds + logger."""
    ensure_dirs()
    ctx = ExperimentContext()
    if seed is not None:
        ctx.seed = seed
    set_global_seeds(ctx.seed)
    log = get_logger()
    log.info(f"seed={ctx.seed} device={ctx.device} max_seq_len={ctx.max_seq_len}")
    return ctx


def prepare_through_data(
    seed: int | None = None,
    do_integrity_checks: bool = False,
    do_extrativa: bool = False,
    do_ner: bool = False,
    do_fakebr: bool = False,
) -> ExperimentContext:
    """setup + data pipeline."""
    ctx = setup_context(seed=seed)
    data_pipeline(
        ctx,
        do_integrity_checks=do_integrity_checks,
        do_extrativa=do_extrativa,
        do_ner=do_ner,
        do_fakebr=do_fakebr,
    )
    return ctx


def prepare_through_preprocessing(
    seed: int | None = None,
    method: str = "base",
    **data_kwargs,
) -> ExperimentContext:
    """data + preprocessing."""
    ctx = prepare_through_data(seed=seed, **data_kwargs)
    preprocessing_pipeline(ctx, method=method)
    return ctx


def prepare_through_features(
    seed: int | None = None,
    extract_embeddings: bool = True,
    fit_tfidf_vec: bool = True,
    batch_size: int = BATCH_SIZE,
    **data_kwargs,
) -> ExperimentContext:
    """data + preprocessing + features (embeddings BERT + TF-IDF + loaders).

    Popula em ``ctx``:
      ``tokenizer``, ``bert_model``, ``token_train/val/test``, ``cls_train/val/test``,
      ``tfidf_vectorizer``, ``X_train/val/test_tfidf``, ``extras['loaders']``,
      ``extras['embed_dim']``.
    """
    ctx = prepare_through_preprocessing(seed=seed, **data_kwargs)

    if extract_embeddings:
        extractor = EmbeddingExtractor(
            device=ctx.device, max_seq_len=ctx.max_seq_len
        )
        ctx.tokenizer = extractor.tokenizer
        ctx.bert_model = extractor.model
        ctx.extras["embedding_extractor"] = extractor
        ctx.extras["embed_dim"] = extractor.embed_dim

        embs = extract_all_embeddings(
            extractor, ctx.X_train_text, ctx.X_val_text, ctx.X_test_text
        )
        for k, v in embs.items():
            setattr(ctx, k, v)

        ctx.extras["loaders"] = make_loaders(
            ctx.token_train, ctx.token_val, ctx.token_test,
            ctx.y_train, ctx.y_val, ctx.y_test,
            batch_size=batch_size,
        )

    if fit_tfidf_vec:
        ctx.tfidf_vectorizer = fit_tfidf(ctx.X_train_text)
        ctx.X_train_tfidf = ctx.tfidf_vectorizer.transform(ctx.X_train_text)
        ctx.X_val_tfidf = ctx.tfidf_vectorizer.transform(ctx.X_val_text)
        ctx.X_test_tfidf = ctx.tfidf_vectorizer.transform(ctx.X_test_text)

    return ctx


__all__ = [
    "setup_context",
    "prepare_through_data",
    "prepare_through_preprocessing",
    "prepare_through_features",
]
