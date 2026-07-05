"""Orquestrador do subpacote `data`: carregamento → integridade.

**Não faz splits.** Os splits acontecem sobre `text_proc` (texto preprocessado),
então a divisão roda dentro do `preprocessing.run_pipeline`. Quem chama
`data.run_pipeline` isolado fica só com `ctx.df` (e opcionais `df_extr`,
`df_fakebr`). Pra obter splits + X_*_text, chame também
`preprocessing.run_pipeline(ctx)` ou use `_pipeline.prepare_through_preprocessing`.
"""

from __future__ import annotations

from .._context import ExperimentContext
from .fakebr_loader import load_fakebr_corpus
from .integrity_checks import (
    length_by_class,
    ner_top_by_class,
    source_class_analysis,
    temporal_analysis,
)
from .loading import load_and_prepare


def run(
    ctx: ExperimentContext,
    do_integrity_checks: bool = True,
    do_ner: bool = False,
    do_extrativa: bool = True,
    do_fakebr: bool = False,
) -> ExperimentContext:
    """Pipeline de dados (sem splits).

    Args:
        ctx: contexto a ser populado.
        do_integrity_checks: roda análises de fonte/comprimento/temporal.
        do_ner: roda NER (lento, ~2 min). Requer spaCy `pt_core_news_sm`.
        do_extrativa: também carrega a variante extrativa.
        do_fakebr: também carrega o Fake.br-Corpus (cross-dataset).

    Returns:
        O mesmo `ctx`, com os campos populados:
        - `ctx.df` = variante abstrativa, deduplicada e com label_enc
        - `ctx.extras['df_extr']` se `do_extrativa`
        - `ctx.extras['df_fakebr']` se `do_fakebr`
        - `ctx.extras['class_names']`, `ctx.extras['label_encoder']`
        - `ctx.extras['src_excl_*']` se `do_integrity_checks`
    """
    df_abst, le, class_names = load_and_prepare("abstrativa")
    ctx.df = df_abst
    ctx.extras["label_encoder"] = le
    ctx.extras["class_names"] = class_names

    if do_extrativa:
        df_extr, _, _ = load_and_prepare("extrativa", encoder=le)
        ctx.extras["df_extr"] = df_extr

    if do_integrity_checks:
        ctx.extras["src_excl_abst"] = source_class_analysis(df_abst, "Abstrativa")
        if do_extrativa:
            ctx.extras["src_excl_extr"] = source_class_analysis(
                ctx.extras["df_extr"], "Extrativa"
            )
        df_abst = length_by_class(df_abst, "Abstrativa")
        ctx.df = df_abst
        if do_extrativa:
            ctx.extras["df_extr"] = length_by_class(ctx.extras["df_extr"], "Extrativa")
        temporal_analysis(df_abst)

    if do_ner:
        import spacy
        nlp = spacy.load("pt_core_news_sm")
        ctx.extras["ner_abst"] = ner_top_by_class(df_abst, nlp, "Abstrativa")

    if do_fakebr:
        try:
            ctx.extras["df_fakebr"] = load_fakebr_corpus()
        except FileNotFoundError as e:
            # Fake.br ausente nao bloqueia o pipeline IID. A avaliacao
            # cross-dataset apenas sera pulada pelos scripts a jusante
            # (08_cross_dataset.py / run_all.py:284 ja checam isso).
            from ..utils.logging_utils import get_logger
            get_logger().warning(
                f"[fakebr] {e} Cross-dataset sera pulado nesta execucao."
            )

    return ctx


__all__ = ["run"]
