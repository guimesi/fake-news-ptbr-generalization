"""Avaliação OOD: FakeRecogna ↔ Fake.br-Corpus.

Encapsula a lógica reutilizável (encoding de labels externos, auto-detecção
de polaridade, diagnóstico de erros). Cada experimento específico (treinar/
avaliar um modelo em cada direção, montar tabela IID×OOD) é composto pelo
script `scripts/08_cross_dataset.py`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from ..config import SEED
from ..preprocessing.text_cleaning import preprocess_base
from ..utils.io_utils import save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
_DEFAULT_LABEL_MAP: dict[str, int] = {
    "fake": 0, "false": 0, "mentira": 0, "0": 0,
    "true": 1, "real": 1, "verdadeira": 1, "verdade": 1, "1": 1,
}


def encode_external_labels(
    raw_labels,
    encoder: LabelEncoder | None = None,
    label_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tenta encoder.transform; se falhar, usa label_map.

    Returns:
        (y_codificado, mask de válidos, pra filtrar textos correspondentes)
    """
    if encoder is not None:
        try:
            y = encoder.transform(raw_labels)
            return np.asarray(y), np.ones(len(y), dtype=bool)
        except ValueError:
            pass
    label_map = label_map or _DEFAULT_LABEL_MAP
    y = np.array(
        [label_map.get(str(lbl).lower().strip(), -1) for lbl in raw_labels]
    )
    mask = y >= 0
    return y[mask], mask


def evaluate_on_external_corpus(
    corpus_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    encoder: LabelEncoder | None = None,
    apply_preprocessing: bool = True,
    auto_detect_polarity: bool = True,
    min_text_length: int = 10,
) -> dict[str, float | str | int]:
    """Avalia `predict_fn` em corpus externo, com auto-flip de polaridade.

    Returns:
        dict com Accuracy, Precision, Recall, F1, N, model (e nota de flip se houver).
    """
    if apply_preprocessing:
        texts = corpus_df[text_col].astype(str).apply(preprocess_base).tolist()
    else:
        texts = corpus_df[text_col].astype(str).tolist()

    raw_labels = corpus_df[label_col].astype(str).values
    y_ext, mask = encode_external_labels(raw_labels, encoder=encoder)
    if len(texts) != len(mask):
        # encoder.transform devolveu tamanho diferente, recalibra
        texts = [t for t, m in zip(texts, mask) if m]
    else:
        texts = [t for t, m in zip(texts, mask) if m]
        y_ext = y_ext[mask] if mask.sum() != len(mask) else y_ext

    keep = [i for i, t in enumerate(texts) if len(t) >= min_text_length]
    texts = [texts[i] for i in keep]
    y_ext = y_ext[keep]

    preds = predict_fn(texts)
    yp = preds.argmax(1) if preds.ndim == 2 else preds

    polarity_note = ""
    polarity_flipped = False
    polarity_acc_direct: float | None = None
    polarity_acc_flipped: float | None = None
    if auto_detect_polarity:
        polarity_acc_direct = float(accuracy_score(y_ext, yp))
        polarity_acc_flipped = float(accuracy_score(y_ext, 1 - yp))
        if polarity_acc_flipped > polarity_acc_direct + 0.05:
            log.warning(
                f"[{model_name}] Inversão de labels detectada: "
                f"direct={polarity_acc_direct:.3f}, "
                f"flipped={polarity_acc_flipped:.3f}. Aplicando flip."
            )
            yp = 1 - yp
            polarity_note = " [labels invertidos auto]"
            polarity_flipped = True

    metrics = {
        "Accuracy": accuracy_score(y_ext, yp),
        "Precision": precision_score(y_ext, yp, average="macro", zero_division=0),
        "Recall": recall_score(y_ext, yp, average="macro", zero_division=0),
        "F1": f1_score(y_ext, yp, average="macro", zero_division=0),
        "N": len(y_ext),
        "model": model_name + polarity_note,
        # Salvaguarda de polaridade: registro auditável de auto-flip.
        "polarity_flipped": polarity_flipped,
        "polarity_acc_direct": polarity_acc_direct,
        "polarity_acc_flipped": polarity_acc_flipped,
        # Predicoes para gerar matriz de confusao downstream.
        "y_true": y_ext,
        "y_pred": yp,
    }
    log.info(
        f'[OOD] {metrics["model"]}: N={metrics["N"]} '
        f'Acc={metrics["Accuracy"]:.4f} F1={metrics["F1"]:.4f} '
        f'polarity_flipped={polarity_flipped}'
    )
    return metrics


def train_on_fakebr_eval_main(
    df_fakebr: pd.DataFrame,
    X_main_test: list[str],
    y_main_test: np.ndarray,
    extractor,
    tokenizer,
    bert_model,
    device: str = "cpu",
    epochs_deep: int = 15,
    epochs_bert: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Cross-dataset INVERSO: treina Ens2(CNN+LSTM) + BERTimbau FT no Fake.br
    e avalia no test set principal (FakeRecogna abstrativa).

    Salva CSV
    `17_cross_dataset_inverse_ood.csv` + CM correspondentes.

    Args:
        df_fakebr: DataFrame Fake.br com colunas 'text', 'label'.
        X_main_test, y_main_test: split de teste do FakeRecogna abstrativa.
        extractor: ctx.extras['embedding_extractor'] (BERTimbau encoder).
        tokenizer: ctx.tokenizer.
        bert_model: ctx.bert_model (pre-treinado, sem head).
        device, epochs_*, seed: hiperparametros.

    Returns:
        DataFrame com [model, Accuracy, Precision, Recall, F1, N].
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from ..features import make_loaders
    from ..models import (
        TextCNN, TextLSTM, train_ensemble_on_variant,
        train_bertimbau_finetune,
    )
    from .confusion_matrices import save_cm

    # 1) Encode labels (str -> int) com encoder fixo (fake=0, real=1)
    le = LabelEncoder().fit(["fake", "real"])
    df_fbr_clean = df_fakebr.dropna(subset=["text", "label"]).copy()
    df_fbr_clean["label_enc"] = le.transform(df_fbr_clean["label"].astype(str).str.lower())

    # 2) Split estratificado 70/10/20 dentro do Fake.br
    Xfbr = df_fbr_clean["text"].astype(str).tolist()
    yfbr = df_fbr_clean["label_enc"].to_numpy()
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        Xfbr, yfbr, test_size=0.30, stratify=yfbr, random_state=seed,
    )
    Xvl, _Xte_fbr, yvl, _yte_fbr = train_test_split(
        Xtmp, ytmp, test_size=2 / 3, stratify=ytmp, random_state=seed,
    )

    # 3) Extrai embeddings com o BERTimbau (mesmo backbone usado no FakeRecogna)
    log.info(f"[cross-inv] Extraindo embeddings Fake.br Tr={len(Xtr)} Vl={len(Xvl)} TeMain={len(X_main_test)}")
    e_tr = extractor.extract_token_embs(Xtr)
    e_vl = extractor.extract_token_embs(Xvl)
    e_te = extractor.extract_token_embs(list(X_main_test))
    ld_tr, ld_vl, ld_te = make_loaders(e_tr, e_vl, e_te, ytr, yvl, y_main_test)

    embed_dim = e_tr.shape[-1] if hasattr(e_tr, "shape") else 768

    # 4) Treina Ens2 (CNN+LSTM) no Fake.br, avalia no test principal
    res_ens = train_ensemble_on_variant(
        TextCNN, TextLSTM,
        dict(embed_dim=embed_dim, num_classes=2, dropout=0.5),
        dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
             num_classes=2, dropout=0.4),
        ld_tr, ld_vl, ld_te, y_main_test,
        device=device, epochs=epochs_deep, suffix="fbr_to_main", seed=seed,
    )
    yp_ens = res_ens["y_pred"]
    save_cm(
        y_main_test, yp_ens, model_name="Ens2 (CNN+LSTM)",
        class_names=["fake", "real"],
        save_as_prefix="cm_ood_inverse_ens2",
        title_suffix="Fake.br -> FakeRecogna",
    )

    # 5) Treina BERTimbau FT no Fake.br, avalia no test principal
    bert_clf_fbr, probs_bert = train_bertimbau_finetune(
        bert_model, tokenizer,
        Xtr, ytr, Xvl, yvl,
        list(X_main_test), y_main_test,
        num_classes=2, epochs=epochs_bert, batch_size=16,
        device=device, seed=seed,
        result_key="BERTimbau FT [Fake.br->Main]",
    )
    yp_bert = probs_bert.argmax(1).numpy()
    save_cm(
        y_main_test, yp_bert, model_name="BERTimbau FT",
        class_names=["fake", "real"],
        save_as_prefix="cm_ood_inverse_bert_ft",
        title_suffix="Fake.br -> FakeRecogna",
    )
    del bert_clf_fbr
    import gc; gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # 6) Tabela
    rows = []
    for name, yp in [
        ("Ens2 (CNN+LSTM) [Fake.br->Main]", yp_ens),
        ("BERTimbau FT [Fake.br->Main]", yp_bert),
    ]:
        rows.append({
            "model": name,
            "Accuracy": float(accuracy_score(y_main_test, yp)),
            "Precision": float(precision_score(y_main_test, yp, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_main_test, yp, average="macro", zero_division=0)),
            "F1": float(f1_score(y_main_test, yp, average="macro", zero_division=0)),
            "N": int(len(y_main_test)),
        })
    df_inv = pd.DataFrame(rows).round(4)
    from ..utils.io_utils import save_table
    save_table(df_inv, "17_cross_dataset_inverse_ood")
    log.info(f"[cross-inv] inverse OOD:\n{df_inv}")
    return df_inv


def error_diagnosis_cross_dataset(
    probs: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Model",
    save_as: str | None = "17_6_cross_dataset_error_diagnosis",
) -> pd.DataFrame:
    """Matriz de confusão + assimetria + confiança nos erros."""
    preds_raw = probs.argmax(1)
    acc_direct = (preds_raw == y_true).mean()
    acc_flipped = (preds_raw == (1 - y_true)).mean()
    flipped = acc_flipped > acc_direct + 0.05
    preds = (1 - preds_raw) if flipped else preds_raw
    if flipped:
        log.info(f"Auto-flip: direct={acc_direct:.3f} flipped={acc_flipped:.3f}")

    cm = confusion_matrix(y_true, preds)
    fn_rate = cm[0, 1] / max(1, cm[0].sum())
    fp_rate = cm[1, 0] / max(1, cm[1].sum())
    print(f"\n=== {model_name} ===")
    print("Matriz de confusão (0=fake, 1=real):")
    print(f"                 pred=0    pred=1")
    print(f"real=0 (fake)    {cm[0,0]:7d}    {cm[0,1]:7d}")
    print(f"real=1 (real)    {cm[1,0]:7d}    {cm[1,1]:7d}")
    print(f"\nTaxa fake→real: {fn_rate:.1%}  |  Taxa real→fake: {fp_rate:.1%}")

    err_mask = preds != y_true
    mean_conf = (
        float(probs[err_mask].max(1).mean()) if err_mask.any() else float("nan")
    )

    row = {
        "cm_fake_fake": int(cm[0, 0]),
        "cm_fake_real": int(cm[0, 1]),
        "cm_real_fake": int(cm[1, 0]),
        "cm_real_real": int(cm[1, 1]),
        "fake_to_real_rate": float(fn_rate),
        "real_to_fake_rate": float(fp_rate),
        "mean_conf_on_errors": mean_conf,
        "flipped_autodetect": bool(flipped),
    }
    df = pd.DataFrame([row])
    if save_as is not None:
        save_table(df, save_as)
    return df


def save_polarity_audit_log(
    ood_results: list[dict],
    save_as: str | None = "17_polarity_audit_log",
) -> pd.DataFrame:
    """Extrai um CSV focado nas decisões da salvaguarda de polaridade.

    Espera receber a lista de dicts retornada por `evaluate_on_external_corpus`.
    Cada linha resume, para uma execução cross-dataset: modelo, N, acurácia
    direta, acurácia sob inversão, se o flip foi aplicado e o delta.
    """
    rows = []
    for r in ood_results:
        if "polarity_flipped" not in r:
            continue
        direct = r.get("polarity_acc_direct")
        flipped = r.get("polarity_acc_flipped")
        delta = (
            float(flipped) - float(direct)
            if direct is not None and flipped is not None
            else None
        )
        rows.append(
            {
                "model": r.get("model", "?"),
                "N": r.get("N"),
                "acc_direct": direct,
                "acc_flipped": flipped,
                "delta_flip_minus_direct": delta,
                "polarity_flipped": r.get("polarity_flipped", False),
            }
        )
    df = pd.DataFrame(rows)
    if save_as is not None and len(df) > 0:
        save_table(df, save_as)
    return df


__all__ = [
    "encode_external_labels",
    "evaluate_on_external_corpus",
    "error_diagnosis_cross_dataset",
    "save_polarity_audit_log",
]
