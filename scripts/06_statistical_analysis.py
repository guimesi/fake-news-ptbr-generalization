"""Etapa 6 — Análise estatística: log-odds (Seção 4) + McNemar (Seção 13.2).

Se chamado isoladamente, retreina os modelos para gerar predições. Use
`--from-04` se já rodou `04_train_deep_models.py` na mesma sessão Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features, prepare_through_preprocessing
from _training import train_bert_classifier, train_deep_ensemble


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logodds-only", action="store_true",
        help="Só roda log-odds + chi² (não precisa treinar modelos).",
    )
    p.add_argument(
        "--no-bert", action="store_true",
        help="Não treina BERTimbau FT (pula essa entrada na McNemar).",
    )
    args = p.parse_args()

    from fakerecogna2.statistics import chi2_top, lex_analysis, mcnemar_pairwise_holm

    if args.logodds_only:
        ctx = prepare_through_preprocessing()
        class_names = ctx.extras.get("class_names", ["fake", "real"])
        lex_analysis(ctx.df, name="abstrativa")
        if "df_extr" in ctx.extras:
            lex_analysis(ctx.extras["df_extr"], name="extrativa")
        chi2_top(ctx.df, class_names=class_names)
        return 0

    # Caminho completo: treina modelos + lex + McNemar
    ctx = prepare_through_features()
    class_names = ctx.extras.get("class_names", ["fake", "real"])
    lex_analysis(ctx.df, name="abstrativa")
    chi2_top(ctx.df, class_names=class_names)

    train_deep_ensemble(ctx)
    if not args.no_bert:
        train_bert_classifier(ctx)

    mcnemar_pairwise_holm(ctx.y_test, ctx.predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
