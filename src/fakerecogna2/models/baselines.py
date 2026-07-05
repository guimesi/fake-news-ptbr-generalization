"""Baselines clássicos: LogReg / LinearSVC / MLP, em TF-IDF e [CLS]."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import SEED
from ..utils.io_utils import RESULTS
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def make_baseline_configs(
    Xtr_tf, Xte_tf, cls_train_np: np.ndarray, cls_test_np: np.ndarray, seed: int = SEED
) -> list[tuple[str, Any, Any, Any]]:
    """6 baselines: 3 sklearn × 2 features (TF-IDF + BERT[CLS])."""
    return [
        (
            "TFIDF+LogReg",
            Xtr_tf,
            Xte_tf,
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        ),
        (
            "TFIDF+SVM",
            Xtr_tf,
            Xte_tf,
            CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
        ),
        (
            "TFIDF+MLP",
            Xtr_tf,
            Xte_tf,
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=50,
                early_stopping=True,
                random_state=seed,
            ),
        ),
        (
            "BERT[CLS]+LogReg",
            cls_train_np,
            cls_test_np,
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        ),
        (
            "BERT[CLS]+SVM",
            cls_train_np,
            cls_test_np,
            CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
        ),
        (
            "BERT[CLS]+MLP",
            cls_train_np,
            cls_test_np,
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=50,
                early_stopping=True,
                random_state=seed,
            ),
        ),
    ]


def train_baselines(
    configs: list[tuple[str, "any", "any", "any"]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    print_table: bool = True,
) -> tuple[dict[str, "any"], dict[str, np.ndarray]]:
    """Treina cada config, registra em RESULTS, retorna (configs_treinados, probs)."""
    trained: list[tuple] = []
    probs: dict[str, np.ndarray] = {}
    for name, Xtr, Xte, mdl in tqdm(configs, desc="Baselines"):
        mdl.fit(Xtr, y_train)
        t0 = time.time()
        yp = mdl.predict(Xte)
        inf = (time.time() - t0) / max(1, len(y_test))
        prob = mdl.predict_proba(Xte) if hasattr(mdl, "predict_proba") else None
        RESULTS[name] = {
            "Accuracy": accuracy_score(y_test, yp),
            "Precision": precision_score(y_test, yp, average="macro", zero_division=0),
            "Recall": recall_score(y_test, yp, average="macro", zero_division=0),
            "F1": f1_score(y_test, yp, average="macro", zero_division=0),
            "Inference (ms)": inf * 1000,
        }
        if prob is not None:
            probs[name] = prob
        trained.append((name, Xtr, Xte, mdl))
        log.info(
            f'{name}: Acc={RESULTS[name]["Accuracy"]:.4f} '
            f'F1={RESULTS[name]["F1"]:.4f}'
        )

    if print_table:
        print(tabulate(pd.DataFrame(RESULTS).T.round(4), headers="keys", tablefmt="github"))
    return trained, probs


# Para evitar warning de Any em type hints quando typing não está importado
Any = object


def train_baselines_multiseed(
    Xtr_tf, Xte_tf,
    cls_train_np: np.ndarray, cls_test_np: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Treina os 6 baselines uma vez por seed, agrega mean±std das metricas.

    Modelos que dependem de random_state (LogReg, MLP) variam por seed;
    LinearSVC dentro de CalibratedClassifierCV varia pelo CV-split que e
    determinado por scikit-learn (nao expoe seed direto).

    Returns:
        DataFrame indexado por modelo com colunas Accuracy/F1/Precision/Recall
        em mean e std.
    """
    from collections import defaultdict
    agg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in seeds:
        configs = make_baseline_configs(Xtr_tf, Xte_tf, cls_train_np, cls_test_np, seed=s)
        for name, Xtr, Xte, mdl in configs:
            mdl.fit(Xtr, y_train)
            yp = mdl.predict(Xte)
            agg[name]["Accuracy"].append(accuracy_score(y_test, yp))
            agg[name]["Precision"].append(precision_score(y_test, yp, average="macro", zero_division=0))
            agg[name]["Recall"].append(recall_score(y_test, yp, average="macro", zero_division=0))
            agg[name]["F1"].append(f1_score(y_test, yp, average="macro", zero_division=0))
        log.info(f"  baselines multi-seed: seed={s} ok")

    rows = []
    for name, metrics in agg.items():
        row = {"Model": name}
        for m, vals in metrics.items():
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).round(6)


__all__ = [
    "make_baseline_configs",
    "train_baselines",
    "train_baselines_multiseed",
]
