"""Etapa 7: explicabilidade (LIME, Integrated Gradients e Attention Rollout).

Treina os modelos neurais e o BERTimbau FT, seleciona exemplos da matriz de
confusão e gera as explicações: LIME para o ensemble Ens3, comparação Integrated
Gradients vs Attention Rollout (salvo `--skip-ig`) e o teste de estabilidade do
LIME por Jaccard (salvo `--skip-stability`).

Observação: este script usa `select_lime_targets` (TP/FN) e não agrega tokens
por TP/TN/FP/FN. A versão estratificada por 4 células (usada no artigo) está no
run_all.py (etapa 9). O `ensemble_predict_proba` interno aqui não é batcheado, o
que pode consumir mais memória em corpora grandes.

Pré-requisitos: nenhum script anterior (treina internamente; o BERTimbau FT é
necessário para IG/Attention). GPU recomendada.

Saídas:
    outputs/metrics/15_lime_records.csv, 15_lime_tokens_by_tag.csv,
    I_lime_stability.csv; figuras em outputs/figures/lime/

Uso:
    python scripts/07_explainability.py
    python scripts/07_explainability.py --n-lime 12 --skip-stability
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import train_bert_classifier, train_deep_ensemble


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-lime", type=int, default=8, help="Nº de exemplos LIME.")
    p.add_argument("--skip-stability", action="store_true", help="Pula Jaccard LIME (lento).")
    p.add_argument("--skip-ig", action="store_true", help="Pula IG/Attention comparison.")
    args = p.parse_args()

    ctx = prepare_through_features()
    train_deep_ensemble(ctx)
    train_bert_classifier(ctx)

    class_names = ctx.extras.get("class_names", ["fake", "real"])
    ens3_preds = ctx.predictions["Ens3 (CNN+LSTM+ConvLSTM)"]

    def ensemble_predict_proba(texts: list[str]) -> np.ndarray:
        enc = ctx.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=ctx.max_seq_len, return_tensors="pt",
        ).to(ctx.device)
        with torch.no_grad():
            embs = ctx.bert_model(**enc).last_hidden_state
        cnn_m = ctx.models["CNN"]; lstm_m = ctx.models["LSTM"]; conv_m = ctx.models["ConvLSTM"]
        cnn_m.eval(); lstm_m.eval(); conv_m.eval()
        with torch.no_grad():
            p = (
                torch.softmax(cnn_m(embs), 1)
                + torch.softmax(lstm_m(embs), 1)
                + torch.softmax(conv_m(embs), 1)
            ) / 3
        return p.cpu().numpy()

    # 15.1 LIME
    from fakerecogna2.explainability import (
        AttentionRollout,
        compare_xai_for_examples,
        lime_stability,
        run_lime_explanations,
        select_lime_targets,
    )
    targets = select_lime_targets(ctx.y_test, ens3_preds, class_names)[: args.n_lime]
    run_lime_explanations(
        ctx.X_test_text, ctx.y_test, ens3_preds, ensemble_predict_proba,
        class_names, targets=targets,
    )

    # 18 IG + Attention Rollout
    if not args.skip_ig:
        compare_xai_for_examples(
            ctx.X_test_text, ctx.y_test, ens3_preds, class_names,
            ctx.bert_clf, ctx.tokenizer, targets=targets, device=ctx.device,
        )

    # I.1 Estabilidade LIME
    if not args.skip_stability:
        lime_stability(
            ctx.X_test_text, ctx.y_test, ens3_preds,
            ensemble_predict_proba, class_names,
        )

    print("\nOK. Plots em outputs/figures/lime/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
