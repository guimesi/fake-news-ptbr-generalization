"""Calibração: ECE, Brier, reliability diagram (cell 43)."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from tabulate import tabulate

from ..config import CALIBRATION_BINS, RELIABILITY_BINS
from ..utils.io_utils import save_plot, save_table




# -- API limpa ----------------------------------------------------------------
def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = CALIBRATION_BINS
) -> float:
    """ECE — distância média entre confidence e accuracy por bin."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_b = (pred[mask] == y[mask]).mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def plot_reliability(
    probs_dict: dict[str, np.ndarray],
    y: np.ndarray,
    save_as: str | None = "13_reliability",
    n_bins: int = RELIABILITY_BINS,
) -> None:
    """Reliability diagram: confidence × accuracy."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for lbl, P in probs_dict.items():
        conf = P.max(1)
        pred = P.argmax(1)
        bins = np.linspace(0, 1, n_bins + 1)
        mids, accs = [], []
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.sum() > 0:
                mids.append(conf[mask].mean())
                accs.append((pred[mask] == y[mask]).mean())
        ax.plot(
            mids,
            accs,
            marker="o",
            label=f"{lbl} (ECE={expected_calibration_error(P, y):.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Reliability Diagram")
    if save_as is not None:
        save_plot(fig, save_as)


def calibration_table(
    probs_dict: dict[str, np.ndarray],
    y: np.ndarray,
    save_as: str | None = "13_calibration",
    print_table: bool = True,
) -> pd.DataFrame:
    """Tabela com Brier (binário) e ECE por modelo."""
    rows = []
    for name, P in probs_dict.items():
        if P.shape[1] == 2:
            b = brier_score_loss(y, P[:, 1])
            ece = expected_calibration_error(P, y)
            rows.append({"Model": name, "Brier": float(b), "ECE": ece})
    df = pd.DataFrame(rows).round(4)
    if print_table:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


__all__ = [
    "expected_calibration_error",
    "plot_reliability",
    "calibration_table",
]
