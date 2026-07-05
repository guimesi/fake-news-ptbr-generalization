"""Wrappers finos sobre sklearn.metrics.

Usa `average='macro'` por padrão.
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score,
    brier_score_loss, precision_recall_curve, average_precision_score,
)
import numpy as np


def standard_report(y_true, y_pred, average: str = "macro") -> dict:
    """Conjunto canônico de métricas."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


__all__ = [
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "classification_report", "confusion_matrix",
    "roc_curve", "auc", "roc_auc_score",
    "brier_score_loss", "precision_recall_curve", "average_precision_score",
    "standard_report",
]
