"""Helpers de treino compartilhados pelos scripts numerados.

Centraliza o treinamento dos modelos deep (CNN, LSTM, ConvLSTM, BERT FT) +
construção dos ensembles. Reusado pelos scripts 04, 05, 07, 08, 09, 11, 12, 13.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fakerecogna2 import ExperimentContext
from fakerecogna2.config import SEEDS_MULTI
from fakerecogna2.models import (
    BERTClassifier,  # noqa: F401  (re-export pra conveniência)
    TextCNN,
    TextConvLSTM,
    TextLSTM,
    aggregate_multiseed_results,
    avg_probs,
    get_val_probs,
    grid_search_2model,
    grid_search_3model,
    register_ensemble,
    train_bertimbau_finetune,
    train_model,
    train_multiseed,
)


def train_deep_ensemble(
    ctx: ExperimentContext,
    seeds: list[int] = SEEDS_MULTI,
    epochs: int = 30,
    grid_epochs: int = 15,
) -> ExperimentContext:
    """Treina CNN/LSTM/ConvLSTM com multi-seed e monta ensembles 2/3-model.

    Popula em ``ctx.models``: 'CNN', 'LSTM', 'ConvLSTM' (instâncias da seed principal).
    Popula em ``ctx.probabilities``: 'CNN', 'LSTM', 'ConvLSTM', 'Ens2 (CNN+LSTM)',
    'Ens3 (CNN+LSTM+ConvLSTM)', 'WEns2 (α=X.XX)', 'WEns3 (a/b/c)'.
    Popula em ``ctx.predictions``: idem.
    Atualiza ``RESULTS`` e ``RESULTS_MULTISEED`` (módulo `utils.io_utils`).
    """
    if "loaders" not in ctx.extras:
        raise RuntimeError(
            "ctx.extras['loaders'] vazio, rode `_pipeline.prepare_through_features` antes."
        )
    train_loader, val_loader, test_loader = ctx.extras["loaders"]
    embed_dim = ctx.extras.get("embed_dim", 768)
    num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))

    cnn_metrics, cnn_probs_list, _ = train_multiseed(
        TextCNN, train_loader, val_loader, test_loader,
        dict(embed_dim=embed_dim, num_classes=num_classes, dropout=0.5),
        seeds=seeds, device=ctx.device, epochs=epochs, model_name="CNN",
    )
    lstm_metrics, lstm_probs_list, _ = train_multiseed(
        TextLSTM, train_loader, val_loader, test_loader,
        dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
             num_classes=num_classes, dropout=0.4),
        seeds=seeds, device=ctx.device, epochs=epochs, model_name="LSTM",
    )
    convlstm_metrics, convlstm_probs_list, _ = train_multiseed(
        TextConvLSTM, train_loader, val_loader, test_loader,
        dict(embed_dim=embed_dim, num_filters=64, hidden_dim=64,
             num_classes=num_classes, dropout=0.4),
        seeds=seeds, device=ctx.device, epochs=epochs, model_name="ConvLSTM",
    )
    aggregate_multiseed_results(cnn_metrics, "CNN")
    aggregate_multiseed_results(lstm_metrics, "LSTM")
    aggregate_multiseed_results(convlstm_metrics, "ConvLSTM")

    cnn_probs = avg_probs(cnn_probs_list)
    lstm_probs = avg_probs(lstm_probs_list)
    convlstm_probs = avg_probs(convlstm_probs_list)
    ctx.probabilities["CNN"] = cnn_probs.numpy()
    ctx.probabilities["LSTM"] = lstm_probs.numpy()
    ctx.probabilities["ConvLSTM"] = convlstm_probs.numpy()
    ctx.predictions["CNN"] = cnn_probs.argmax(1).numpy()
    ctx.predictions["LSTM"] = lstm_probs.argmax(1).numpy()
    ctx.predictions["ConvLSTM"] = convlstm_probs.argmax(1).numpy()

    # --- Consistencia RESULTS <-> predicoes (correcao de causa-raiz) ---
    # `aggregate_multiseed_results` grava em RESULTS[name] a metrica da seed
    # principal (metrics_list[0] = modelo unico). Mas as predicoes canonicas
    # usadas por matrizes de confusao, bootstrap CI e Tabela 1 do artigo sao o
    # ensemble de probabilidades MEDIAS das sementes (ctx.predictions[name],
    # via avg_probs acima). Para garantir 16_final_results.csv == cm_iid_* ==
    # 13_bootstrap_ci, recomputamos o escalar de RESULTS a partir de
    # ctx.predictions (preservando o tempo de inferencia medido na seed
    # principal). As medias por-seed (modelo unico) permanecem apenas em
    # RESULTS_MULTISEED / iid_multiseed.csv como medida de estabilidade.
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )
    from fakerecogna2.utils.io_utils import RESULTS

    for _name, _m0 in (
        ("CNN", cnn_metrics[0]),
        ("LSTM", lstm_metrics[0]),
        ("ConvLSTM", convlstm_metrics[0]),
    ):
        _yp = ctx.predictions[_name]
        RESULTS[_name] = {
            "Accuracy": float(accuracy_score(ctx.y_test, _yp)),
            "Precision": float(precision_score(ctx.y_test, _yp, average="macro", zero_division=0)),
            "Recall": float(recall_score(ctx.y_test, _yp, average="macro", zero_division=0)),
            "F1": float(f1_score(ctx.y_test, _yp, average="macro", zero_division=0)),
            "Inference (ms)": _m0["Inference (ms)"],
        }

    # Refita um modelo da seed principal pra grid search (precisa val_probs)
    def _fit(cls, kwargs, seed):
        m = cls(**kwargs)
        m, _ = train_model(
            m, train_loader, val_loader, device=ctx.device, epochs=grid_epochs,
            model_name=f"αgrid_{cls.__name__}", seed=seed,
        )
        return m, get_val_probs(m, val_loader, device=ctx.device)

    cnn_m, vp_cnn = _fit(TextCNN, dict(embed_dim=embed_dim, num_classes=num_classes, dropout=0.5), 42)
    lstm_m, vp_lstm = _fit(TextLSTM, dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                                          num_classes=num_classes, dropout=0.4), 42)
    conv_m, vp_conv = _fit(TextConvLSTM, dict(embed_dim=embed_dim, num_filters=64, hidden_dim=64,
                                              num_classes=num_classes, dropout=0.4), 42)
    ctx.models["CNN"] = cnn_m
    ctx.models["LSTM"] = lstm_m
    ctx.models["ConvLSTM"] = conv_m

    y_val = ctx.y_val
    best_a2, _ = grid_search_2model(vp_cnn, vp_lstm, y_val)
    best_abc, _ = grid_search_3model(vp_cnn, vp_lstm, vp_conv, y_val)

    ens2_probs = (cnn_probs + lstm_probs) / 2
    ens3_probs = (cnn_probs + lstm_probs + convlstm_probs) / 3
    w2_probs = best_a2 * cnn_probs + (1 - best_a2) * lstm_probs
    w3_probs = (
        best_abc[0] * cnn_probs + best_abc[1] * lstm_probs + best_abc[2] * convlstm_probs
    )

    y_test = ctx.y_test
    ctx.predictions["Ens2 (CNN+LSTM)"] = register_ensemble(
        "Ens2 (CNN+LSTM)", ens2_probs, y_test
    )
    ctx.predictions["Ens3 (CNN+LSTM+ConvLSTM)"] = register_ensemble(
        "Ens3 (CNN+LSTM+ConvLSTM)", ens3_probs, y_test
    )
    ctx.predictions[f"WEns2 (α={best_a2:.2f})"] = register_ensemble(
        f"WEns2 (α={best_a2:.2f})", w2_probs, y_test
    )
    ctx.predictions[
        f"WEns3 ({best_abc[0]:.1f}/{best_abc[1]:.1f}/{best_abc[2]:.1f})"
    ] = register_ensemble(
        f"WEns3 ({best_abc[0]:.1f}/{best_abc[1]:.1f}/{best_abc[2]:.1f})",
        w3_probs, y_test,
    )
    ctx.probabilities["Ens2 (CNN+LSTM)"] = ens2_probs.numpy()
    ctx.probabilities["Ens3 (CNN+LSTM+ConvLSTM)"] = ens3_probs.numpy()
    # Probabilidades dos ensembles ponderados (para ECE estendido).
    ctx.probabilities[f"WEns2 (α={best_a2:.2f})"] = w2_probs.numpy()
    ctx.probabilities[
        f"WEns3 ({best_abc[0]:.1f}/{best_abc[1]:.1f}/{best_abc[2]:.1f})"
    ] = w3_probs.numpy()

    return ctx


def train_bert_classifier(
    ctx: ExperimentContext,
    epochs: int = 10,
    batch_size: int = 16,
    seeds: list[int] | None = None,
) -> ExperimentContext:
    """Treina BERTimbau fine-tuned. Popula ctx.bert_clf, ctx.probabilities['BERTimbau FT'].

    Args:
        seeds: se fornecido com 2+ sementes, treina multi-seed; agrega
            mean+/-std em RESULTS_MULTISEED e usa a seed principal (a
            primeira) para o classificador downstream.
    """
    if ctx.bert_model is None or ctx.tokenizer is None:
        raise RuntimeError(
            "ctx.bert_model/tokenizer vazios, rode `_pipeline.prepare_through_features` antes."
        )
    num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))
    seeds_list = seeds or [ctx.seed]
    multi_seed = len(seeds_list) > 1

    if not multi_seed:
        bert_clf, probs = train_bertimbau_finetune(
            ctx.bert_model, ctx.tokenizer,
            ctx.X_train_text, ctx.y_train,
            ctx.X_val_text, ctx.y_val,
            ctx.X_test_text, ctx.y_test,
            num_classes=num_classes,
            epochs=epochs, batch_size=batch_size,
            max_seq_len=ctx.max_seq_len, device=ctx.device,
            seed=ctx.seed,
        )
    else:
        # Multi-seed: treina N modelos, mantem o primeiro para downstream
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import pandas as pd
        from fakerecogna2.utils.io_utils import RESULTS_MULTISEED, save_table

        bert_clf_principal = None
        probs_principal = None
        per_seed_metrics: list[dict] = []
        for i, sd in enumerate(seeds_list):
            clf_sd, probs_sd = train_bertimbau_finetune(
                ctx.bert_model, ctx.tokenizer,
                ctx.X_train_text, ctx.y_train,
                ctx.X_val_text, ctx.y_val,
                ctx.X_test_text, ctx.y_test,
                num_classes=num_classes,
                epochs=epochs, batch_size=batch_size,
                max_seq_len=ctx.max_seq_len, device=ctx.device,
                seed=sd,
                result_key=f"BERTimbau FT [seed={sd}]",
            )
            yp = probs_sd.argmax(1).numpy()
            per_seed_metrics.append({
                "seed": sd,
                "Accuracy": float(accuracy_score(ctx.y_test, yp)),
                "Precision": float(precision_score(ctx.y_test, yp, average="macro", zero_division=0)),
                "Recall": float(recall_score(ctx.y_test, yp, average="macro", zero_division=0)),
                "F1": float(f1_score(ctx.y_test, yp, average="macro", zero_division=0)),
            })
            if i == 0:
                bert_clf_principal, probs_principal = clf_sd, probs_sd
            else:
                del clf_sd
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Agrega
        df_ms = pd.DataFrame(per_seed_metrics)
        agg = {
            metric: (
                float(df_ms[metric].mean()),
                float(df_ms[metric].std(ddof=1)) if len(df_ms) > 1 else 0.0,
            )
            for metric in ("Accuracy", "Precision", "Recall", "F1")
        }
        RESULTS_MULTISEED["BERTimbau FT"] = agg
        save_table(df_ms.round(6), "bertimbau_ft_multiseed")
        bert_clf, probs = bert_clf_principal, probs_principal

    ctx.bert_clf = bert_clf
    ctx.probabilities["BERTimbau FT"] = probs.numpy()
    ctx.predictions["BERTimbau FT"] = probs.argmax(1).numpy()
    return ctx


def make_predict_ens3(ctx: ExperimentContext, batch_size: int = 32):
    """Cria uma função `predict_fn(texts) -> np.ndarray` *batched* do ensemble
    CNN+LSTM+ConvLSTM. Necessário pra evitar OOM em corpora grandes
    (cross-dataset, adversarial). Usa o backbone BERTimbau do `ctx`.
    """
    if not all(k in ctx.models for k in ("CNN", "LSTM", "ConvLSTM")):
        raise RuntimeError(
            "Modelos do ensemble não encontrados em ctx.models, "
            "rode `train_deep_ensemble(ctx)` antes."
        )
    if ctx.bert_model is None or ctx.tokenizer is None:
        raise RuntimeError("ctx.bert_model/tokenizer vazios")

    def predict_ens3(texts) -> np.ndarray:
        texts = list(texts)
        all_probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = ctx.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=ctx.max_seq_len,
                return_tensors="pt",
            ).to(ctx.device)
            with torch.no_grad():
                embs = ctx.bert_model(**enc).last_hidden_state
                ms = [ctx.models[k].eval() for k in ("CNN", "LSTM", "ConvLSTM")]
                p = sum(torch.softmax(m(embs), 1) for m in ms) / 3
            all_probs.append(p.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return predict_ens3


def make_predict_bert_ft(ctx: ExperimentContext, batch_size: int = 32):
    """Cria predict_fn *batched* do BERTimbau fine-tuned."""
    if ctx.bert_clf is None or ctx.tokenizer is None:
        raise RuntimeError(
            "ctx.bert_clf/tokenizer vazios, rode `train_bert_classifier(ctx)` antes."
        )

    def predict_bert_ft(texts) -> np.ndarray:
        texts = list(texts)
        ctx.bert_clf.eval()
        all_probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = ctx.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=ctx.max_seq_len,
                return_tensors="pt",
            ).to(ctx.device)
            with torch.no_grad():
                logits = ctx.bert_clf(enc["input_ids"], enc["attention_mask"])
            all_probs.append(torch.softmax(logits, 1).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return predict_bert_ft


__all__ = [
    "train_deep_ensemble",
    "train_bert_classifier",
    "make_predict_ens3",
    "make_predict_bert_ft",
]
