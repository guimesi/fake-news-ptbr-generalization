"""Etapa 5 — Avaliação robusta: bootstrap, calibração, CV, stress tests, tabela final.

Cobre Seções 13 (bootstrap, McNemar, calibração, 5-fold CV), 14 (stress tests
fonte/NER/temporal) e 16 (tabela consolidada + gráfico + classification report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import train_bert_classifier, train_deep_ensemble


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--no-bert", action="store_true")
    p.add_argument("--no-cv", action="store_true", help="Pula 5-fold CV (caro).")
    p.add_argument("--skip-stress", action="store_true", help="Pula stress tests.")
    args = p.parse_args()

    ctx = prepare_through_features(do_integrity_checks=False)
    train_deep_ensemble(ctx, epochs=args.epochs)
    if not args.no_bert:
        train_bert_classifier(ctx)

    # 13.1 Bootstrap CI
    from fakerecogna2.evaluation import (
        bootstrap_table,
        calibration_table,
        cross_validate_ensemble,
        plot_reliability,
    )
    from fakerecogna2.reports import (
        build_final_table,
        print_classification_report,
    )
    from fakerecogna2.evaluation import plot_final_comparison
    from fakerecogna2.statistics import mcnemar_pairwise_holm

    bootstrap_table(ctx.y_test, ctx.predictions)

    # 13.2 McNemar
    mcnemar_pairwise_holm(ctx.y_test, ctx.predictions)

    # 13.3 Calibração
    probs_for_cal = {
        k: v for k, v in ctx.probabilities.items()
        if k in ("CNN", "LSTM", "Ens3 (CNN+LSTM+ConvLSTM)", "BERTimbau FT")
    }
    plot_reliability(probs_for_cal, ctx.y_test)
    calibration_table(probs_for_cal, ctx.y_test)

    # 13.4 5-fold CV
    if not args.no_cv:
        import torch
        from fakerecogna2.models import TextCNN, TextLSTM

        all_emb = torch.cat([ctx.token_train, ctx.token_val, ctx.token_test], 0)
        import numpy as np
        all_y = np.concatenate([ctx.y_train, ctx.y_val, ctx.y_test])
        all_texts = list(ctx.X_train_text) + list(ctx.X_val_text) + list(ctx.X_test_text)
        embed_dim = ctx.extras.get("embed_dim", 768)
        num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))
        cross_validate_ensemble(
            all_emb, all_y, all_texts,
            model_classes=[
                (TextCNN, dict(embed_dim=embed_dim, num_classes=num_classes, dropout=0.5), "CNN"),
                (TextLSTM, dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                                num_classes=num_classes, dropout=0.4), "LSTM"),
            ],
            device=ctx.device,
        )

    # 14 Stress tests (chamadas específicas via funções do subpacote)
    if not args.skip_stress:
        from fakerecogna2.evaluation.stress_tests import (
            build_masked_test_loader,
            compare_split_results,
            evaluate_models_on_loader,
            ner_ablation_table,
        )
        from fakerecogna2.utils.io_utils import RESULTS

        # NER masking
        extractor = ctx.extras["embedding_extractor"]
        masked_loader = build_masked_test_loader(
            ctx.X_test_text, ctx.y_test, extractor
        )
        preds_masked = evaluate_models_on_loader(
            ctx.models, masked_loader, device=ctx.device
        )
        preds_orig = {k: ctx.predictions[k] for k in ctx.models.keys()}
        ner_ablation_table(ctx.y_test, preds_orig, preds_masked)

        # Split por fonte/temporal — se houver
        if "splits_source" in ctx.extras:
            # TODO: rodar ensemble no source split — requer re-extração de embeddings
            pass
        if "splits_temporal" in ctx.extras:
            # TODO: rodar ensemble no temporal split — idem
            pass

    # 16 Tabela final + comparativo + classification report
    build_final_table()
    plot_final_comparison()
    ens3_preds = ctx.predictions.get("Ens3 (CNN+LSTM+ConvLSTM)")
    class_names = ctx.extras.get("class_names", ["fake", "real"])
    if ens3_preds is not None:
        print_classification_report(ctx.y_test, ens3_preds, class_names, "Ens3")

    print(f"\nOK. Tabela final em outputs/metrics/16_final_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
