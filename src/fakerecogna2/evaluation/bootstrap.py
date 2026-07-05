"""Bootstrap CI para F1 e Accuracy."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

from ..config import BOOTSTRAP_ITERS, SEED
from ..utils.io_utils import save_table




# -- API limpa ----------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable,
    n_boot: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
    percentile: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Bootstrap percentile CI para um `metric(y_true, y_pred)`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        scores.append(metric(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, percentile[0])), float(np.percentile(scores, percentile[1]))


def f1_macro(a, b) -> float:
    return float(f1_score(a, b, average="macro", zero_division=0))


def bootstrap_table(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_boot: int = BOOTSTRAP_ITERS,
    save_as: str | None = "13_bootstrap_ci",
    print_table: bool = True,
) -> pd.DataFrame:
    """Aplica `bootstrap_ci` em F1 e Acc pra cada modelo. Retorna df."""
    y = np.asarray(y_test)
    rows = []
    for name, yp in predictions.items():
        yp = np.asarray(yp)
        f_lo, f_hi = bootstrap_ci(y, yp, f1_macro, n_boot=n_boot)
        a_lo, a_hi = bootstrap_ci(y, yp, accuracy_score, n_boot=n_boot)
        rows.append(
            {
                "Model": name,
                "F1": f1_macro(y, yp),
                "F1_95ci_lo": f_lo,
                "F1_95ci_hi": f_hi,
                "Acc": float(accuracy_score(y, yp)),
                "Acc_95ci_lo": a_lo,
                "Acc_95ci_hi": a_hi,
            }
        )
    df = pd.DataFrame(rows).round(4)
    if print_table:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


__all__ = [
    "bootstrap_ci",
    "f1_macro",
    "bootstrap_table",
]
