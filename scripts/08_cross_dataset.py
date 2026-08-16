"""Etapa 8 — Avaliação cross-dataset (FakeRecogna ↔ Fake.br-Corpus).

Cobre Seções 17 (OOD direto) e G/H (reverso, por categoria).
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
    p.add_argument("--no-bert", action="store_true")
    args = p.parse_args()

    ctx = prepare_through_features(do_fakebr=True)
    train_deep_ensemble(ctx)
    if not args.no_bert:
        train_bert_classifier(ctx)

    if "df_fakebr" not in ctx.extras:
        print("Fake.br não carregado. Ajuste `data.fakebr_local_path` em configs/config.yaml.")
        return 1

    from fakerecogna2.evaluation import (
        error_diagnosis_cross_dataset,
        evaluate_on_external_corpus,
        save_polarity_audit_log,
    )
    from _training import make_predict_bert_ft, make_predict_ens3

    predict_ens3 = make_predict_ens3(ctx, batch_size=32)
    predict_bert_ft = (
        make_predict_bert_ft(ctx, batch_size=32) if ctx.bert_clf is not None else predict_ens3
    )

    df_fbr = ctx.extras["df_fakebr"]
    encoder = ctx.extras.get("label_encoder")

    results = []
    for name, fn in [("Ens3", predict_ens3), ("BERTimbau FT", predict_bert_ft)]:
        if name == "BERTimbau FT" and ctx.bert_clf is None:
            continue
        results.append(
            evaluate_on_external_corpus(
                df_fbr, "text", "label", name, fn, encoder=encoder
            )
        )

    import pandas as pd
    from fakerecogna2.utils.io_utils import save_table

    df_ood = pd.DataFrame(results)
    save_table(df_ood, "17_cross_dataset_ood")
    print(df_ood.to_string(index=False))

    # Audit trail da salvaguarda de polaridade (Seção 4.7).
    save_polarity_audit_log(results)

    # 17.6 Diagnóstico de erros — usando BERT FT (ou ens3 como fallback).
    # Mesmo pré-processamento e mesma convenção de rótulos (0=real, 1=fake)
    # da avaliação oficial em evaluate_on_external_corpus.
    from fakerecogna2.preprocessing.text_cleaning import preprocess_base

    probs_fn = predict_bert_ft if ctx.bert_clf is not None else predict_ens3
    texts_fbr = df_fbr["text"].astype(str).apply(preprocess_base).tolist()
    probs_fbr = probs_fn(texts_fbr)
    y_fbr = np.array([{"real": 0, "fake": 1}.get(str(l).lower(), -1) for l in df_fbr["label"]])
    mask = y_fbr >= 0
    error_diagnosis_cross_dataset(
        probs_fbr[mask], y_fbr[mask],
        model_name="BERTimbau FT" if ctx.bert_clf is not None else "Ens3",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
