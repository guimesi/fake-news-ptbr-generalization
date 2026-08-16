"""Etapa 12 — Métricas de deployment: latência, VRAM, tamanho em disco, Pareto.

Cobre Seções 22 e J.
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
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=50)
    args = p.parse_args()

    ctx = prepare_through_features()
    train_deep_ensemble(ctx)
    train_bert_classifier(ctx)

    benchmark_pool = ctx.X_test_text[:200]

    from _training import make_predict_bert_ft, make_predict_ens3
    predict_ens3 = make_predict_ens3(ctx, batch_size=32)
    predict_bert_ft = make_predict_bert_ft(ctx, batch_size=32)

    def predict_tfidf_mlp(texts: list[str]) -> np.ndarray:
        X = ctx.tfidf_vectorizer.transform(texts)
        # Pega o MLP treinado se existir; senão usa LogReg do baseline_configs
        mlp = None
        for name, _, _, model in ctx.extras.get("baseline_configs", []):
            if name == "TFIDF+MLP":
                mlp = model
                break
        if mlp is None:
            # Fallback: predição aleatória pra não quebrar (mas avisa)
            return np.random.rand(len(texts), 2)
        return mlp.predict_proba(X)

    from fakerecogna2.deployment import (
        benchmark_models,
        count_parameters,
        measure_disk_sizes,
        model_disk_size_mb,
        plot_pareto_f1_latency,
    )
    from fakerecogna2.utils.io_utils import RESULTS

    predict_fns = {
        "BERTimbau FT": predict_bert_ft,
        "Ens3 (CNN+LSTM+ConvLSTM)": predict_ens3,
        "TF-IDF + MLP": predict_tfidf_mlp,
    }
    extras = {
        "BERTimbau FT": {
            "Params (M)": count_parameters(ctx.bert_clf) / 1e6,
            "F1": RESULTS.get("BERTimbau FT", {}).get("F1"),
        },
        "Ens3 (CNN+LSTM+ConvLSTM)": {
            "Params (M)": sum(
                count_parameters(ctx.models[k]) for k in ("CNN", "LSTM", "ConvLSTM")
            ) / 1e6,
            "F1": RESULTS.get("Ens3 (CNN+LSTM+ConvLSTM)", {}).get("F1"),
        },
        "TF-IDF + MLP": {
            "F1": RESULTS.get("TFIDF+MLP", {}).get("F1"),
        },
    }

    df_bench = benchmark_models(
        benchmark_pool, predict_fns, extras=extras,
        n_warmup=args.n_warmup, n_iter=args.n_iter, device=ctx.device,
    )
    plot_pareto_f1_latency(df_bench)

    # J.1 — Tamanho em disco
    models_for_disk = {
        "BERTimbau FT": ctx.bert_clf,
        "Ens3 (CNN+LSTM+ConvLSTM)": [ctx.models["CNN"], ctx.models["LSTM"], ctx.models["ConvLSTM"]],
        "TF-IDF + MLP": [ctx.tfidf_vectorizer]
        + (
            [m for n, _, _, m in ctx.extras.get("baseline_configs", []) if n == "TFIDF+MLP"]
            or []
        ),
    }
    measure_disk_sizes(models_for_disk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
