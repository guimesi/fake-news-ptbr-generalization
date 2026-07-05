"""Testa o módulo `fakerecogna2.evaluation.metrics` (standard_report)."""

import numpy as np
import pytest

from fakerecogna2.evaluation import metrics


def test_standard_report_perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = y_true.copy()
    r = metrics.standard_report(y_true, y_pred)
    assert r["accuracy"] == pytest.approx(1.0)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)


def test_standard_report_random_split():
    y_true = [0] * 5 + [1] * 5
    y_pred = [0] * 3 + [1] * 7
    r = metrics.standard_report(y_true, y_pred)
    assert 0.0 < r["accuracy"] < 1.0
    assert 0.0 < r["f1"] < 1.0


def test_metrics_module_reexports_sklearn():
    # Sanidade: as funções importadas direto do sklearn devem estar disponíveis.
    for name in [
        "accuracy_score", "precision_score", "recall_score", "f1_score",
        "classification_report", "confusion_matrix", "roc_auc_score",
        "brier_score_loss",
    ]:
        assert hasattr(metrics, name), f"metrics não expõe {name}"
