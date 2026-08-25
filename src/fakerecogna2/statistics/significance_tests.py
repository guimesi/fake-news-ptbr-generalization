"""McNemar pairwise com correção Holm (Seção 13.2 do notebook, célula 42)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from tabulate import tabulate

from ..utils.io_utils import save_table

# Família PRÉ-DEFINIDA do protocolo de McNemar (Cap. 4 §4.4 da dissertação):
# uma configuração por representação de interesse — 7 modelos, 21 pares.
# PLMs/WEns/BERT[CLS] ficam fora por definição da família. Todo caller que
# grave o CSV oficial (13_mcnemar_pairwise_holm) deve filtrar as predições
# por esta tupla antes de chamar `mcnemar_pairwise_holm`.
MCNEMAR_FAMILY: tuple[str, ...] = (
    "CNN", "LSTM", "ConvLSTM",
    "Ens2 (CNN+LSTM)", "Ens3 (CNN+LSTM+ConvLSTM)",
    "BERTimbau FT", "TFIDF+MLP",
)


# -- API limpa ----------------------------------------------------------------
def mcnemar_pair(
    y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray
) -> tuple[float, int, int]:
    """Teste de McNemar entre dois classificadores. Retorna (p, a_only, b_only)."""
    y_true = np.asarray(y_true)
    ca, cb = preds_a == y_true, preds_b == y_true
    tab = [
        [int((ca & cb).sum()), int((ca & ~cb).sum())],
        [int((~ca & cb).sum()), int((~ca & ~cb).sum())],
    ]
    return mcnemar(tab, exact=True).pvalue, tab[0][1], tab[1][0]


def mcnemar_pairwise_holm(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    save_as: str | None = "13_mcnemar_pairwise_holm",
    alpha: float = 0.05,
    print_table: bool = True,
) -> pd.DataFrame:
    """McNemar entre todos os pares + correção Holm-Bonferroni.

    Args:
        y_true: rótulos verdadeiros do teste.
        predictions: dict modelo→array de predições.
        save_as: nome do arquivo CSV em outputs/metrics/ (None pra não salvar).
        alpha: nível de significância.
        print_table: imprime tabela com tabulate.

    Returns:
        DataFrame ordenado por p ascendente: A, B, A_only_correct, B_only_correct,
        p, p_holm, sig@<alpha>.
    """
    pairs = list(combinations(predictions.keys(), 2))
    rows = []
    for a, b in pairs:
        p, a_b, b_a = mcnemar_pair(y_true, predictions[a], predictions[b])
        rows.append(
            {"A": a, "B": b, "A_only_correct": a_b, "B_only_correct": b_a, "p": p}
        )
    df_mc = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df_mc)
    # Holm step-down completo: (m-i)*p com máximo cumulativo (monotonicidade).
    adj = (m - df_mc.index.values) * df_mc["p"].values
    df_mc["p_holm"] = np.minimum(np.maximum.accumulate(adj), 1.0)
    df_mc[f"sig@{alpha}"] = df_mc["p_holm"] < alpha

    if print_table:
        print(tabulate(df_mc.round(6), headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df_mc, save_as)
    return df_mc


__all__ = [
    "mcnemar_pair",
    "mcnemar_pairwise_holm",
]
