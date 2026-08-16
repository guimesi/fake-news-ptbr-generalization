"""Matrizes de confusao + metricas por classe.

Gera para cada modelo:
  - cm_<save_as>.csv: tabela 2x2 + linha de totais + metricas por classe
  - cm_<save_as>.png: heatmap da matriz de confusao

Atende as declaracoes da Secao 5.3 e 5.6 do Cap. 5 da dissertacao, que
prometem incluir matrizes de confusao e metricas por classe na versao final.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..utils.io_utils import save_plot, save_table
from ..utils.logging_utils import get_logger

log = get_logger()


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Precision/Recall/F1/Support por classe.

    Retorna DataFrame com uma linha por classe + linha 'macro avg'.
    """
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.bincount(y_true.astype(int), minlength=len(class_names))
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "Classe": name,
            "Precision": round(float(p[i]), 4),
            "Recall":    round(float(r[i]), 4),
            "F1":        round(float(f[i]), 4),
            "Support":   int(support[i]),
        })
    rows.append({
        "Classe": "macro avg",
        "Precision": round(float(p.mean()), 4),
        "Recall":    round(float(r.mean()), 4),
        "F1":        round(float(f.mean()), 4),
        "Support":   int(support.sum()),
    })
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_as: str,
    normalize: bool = False,
) -> None:
    """Plota e salva heatmap da matriz de confusao."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    data = cm.astype(float)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = data / data.sum(axis=1, keepdims=True)
            data = np.nan_to_num(data, nan=0.0)
        fmt_cell = lambda v: f"{v:.3f}"
    else:
        fmt_cell = lambda v: f"{int(v):d}"

    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    # Annotate
    thr = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt_cell(cm[i, j] if not normalize else data[i, j]),
                    ha="center", va="center",
                    color="white" if data[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_plot(fig, save_as)


def save_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: list[str],
    save_as_prefix: str,
    title_suffix: str = "",
) -> dict[str, float]:
    """Computa CM + metricas por classe e salva CSV + PNG.

    Args:
        save_as_prefix: prefixo do nome de arquivo (sem extensao).
            Sufixos `_cm` (CSV+PNG) e `_per_class` (CSV) sao apendados.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # CSV da matriz
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )
    cm_df["Total_true"] = cm.sum(axis=1)
    totals = list(cm.sum(axis=0)) + [int(cm.sum())]
    cm_df.loc["Total_pred"] = totals
    save_table(cm_df, f"{save_as_prefix}_cm")

    # CSV de metricas por classe
    pc_df = per_class_metrics(y_true, y_pred, class_names)
    save_table(pc_df, f"{save_as_prefix}_per_class")

    # PNG do heatmap
    title = f"Matriz de confusao - {model_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    plot_confusion_matrix(cm, class_names, title, save_as=f"{save_as_prefix}_cm")

    log.info(f"[CM] {model_name}: saved {save_as_prefix}_cm.csv/png + _per_class.csv")
    return {
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else None,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else None,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else None,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else None,
    }


def save_all_cms(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    class_names: list[str],
    prefix: str = "cm_iid",
    title_suffix: str = "split IID",
) -> pd.DataFrame:
    """Para cada modelo em `predictions`, gera CM + per-class.

    Retorna DataFrame indice com (modelo, tn, fp, fn, tp).
    """
    rows = []
    for name, y_pred in predictions.items():
        slug = (
            name.lower()
            .replace(" ", "_").replace("+", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_").replace(",", "")
            .replace("=", "").replace("→", "to")
            .replace(":", "").replace(".", "")
            .replace("[", "").replace("]", "")
            .replace("α", "alpha")
        )
        save_as = f"{prefix}_{slug}"
        cells = save_cm(
            y_true=y_true,
            y_pred=y_pred,
            model_name=name,
            class_names=class_names,
            save_as_prefix=save_as,
            title_suffix=title_suffix,
        )
        rows.append({"Modelo": name, **cells, "saved_as": save_as})
    df_idx = pd.DataFrame(rows)
    save_table(df_idx, f"{prefix}_index")
    return df_idx


__all__ = [
    "per_class_metrics",
    "plot_confusion_matrix",
    "save_cm",
    "save_all_cms",
]
