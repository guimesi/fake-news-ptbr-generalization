"""Gráfico comparativo final de modelos."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.io_utils import RESULTS, save_plot




# -- API limpa ----------------------------------------------------------------
def plot_final_comparison(
    results: dict | None = None,
    save_as: str | None = "16_final_comparison",
    cols: tuple[str, ...] = ("Accuracy", "Precision", "Recall", "F1"),
    xlim: tuple[float, float] = (0.80, 1.02),
    colors: tuple[str, ...] = ("#1976D2", "#388E3C", "#F57C00", "#C2185B"),
) -> None:
    """Barras horizontais comparando todos os modelos em RESULTS."""
    results = results if results is not None else RESULTS
    df = pd.DataFrame(results).T[list(cols)].sort_values("F1")
    fig, ax = plt.subplots(figsize=(14, max(5, 0.4 * len(df))))
    x = np.arange(len(df))
    w = 0.2
    for i, (c, col) in enumerate(zip(df.columns, colors)):
        ax.barh(x + i * w, df[c], w, label=c, color=col, alpha=0.88)
    ax.set_yticks(x + 1.5 * w)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Comparação Final de Modelos", fontweight="bold")
    ax.set_xlim(*xlim)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_as is not None:
        save_plot(fig, save_as)


__all__ = [
    "plot_final_comparison",
]
