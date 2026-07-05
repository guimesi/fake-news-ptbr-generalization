"""Etapa 6: análise estatística (log-odds de Dirichlet, Chi quadrado e McNemar).

Com `--logodds-only`, roda apenas a análise lexical (log-odds com prior de
Dirichlet, Monroe et al. 2008, e Chi quadrado sobre TF-IDF), sem treinar
modelos. Sem essa flag, treina os modelos neurais (e o BERTimbau FT, salvo
`--no-bert`) e aplica o teste de McNemar pareado com correção de Holm sobre as
predições.

Pré-requisitos: nenhum script anterior (refaz o setup; no caminho completo
retreina os modelos na própria sessão).

Saídas (outputs/metrics/):
    04_logodds_abstrativa_n<n>_top_<classe>.csv, 04_chi2_top_abstrativa.csv,
    13_mcnemar_pairwise_holm.csv (apenas no caminho completo)

Uso:
    python scripts/06_statistical_analysis.py --logodds-only  # só análise lexical
    python scripts/06_statistical_analysis.py                 # lexical + McNemar
    python scripts/06_statistical_analysis.py --no-bert       # sem o BERTimbau FT
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
