"""Análise de erros: por classe, por comprimento, por fonte."""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

from ..utils.io_utils import save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def error_summary(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    model_name: str = "Model",
) -> None:
    """Loga taxa de erro total e por classe."""
    miss = predictions != y_test
    log.info(
        f"{model_name}: {miss.sum()} erros em {len(y_test)} "
        f"({100*miss.mean():.2f}%)"
    )
    for ci, cn in enumerate(class_names):
        mk = y_test == ci
        n_err = int((miss & mk).sum())
        n_cls = int(mk.sum())
        rate = 100 * n_err / max(1, n_cls)
        log.info(f"  {cn}: {n_err}/{n_cls} erros ({rate:.2f}%)")


def performance_by_length(
    X_test: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    save_as: str | None = "15_performance_by_length",
    print_table: bool = True,
) -> pd.DataFrame:
    """Acurácia/F1 por faixa de comprimento (quartis: Curto/Médio-Curto/Médio-Longo/Longo)."""
    tl = np.array([len(t.split()) for t in X_test])
    qs = np.percentile(tl, [25, 50, 75])
    bins = [0, qs[0], qs[1], qs[2], tl.max() + 1]
    names = ["Curto", "Médio-Curto", "Médio-Longo", "Longo"]
    rows = []
    for i, nm in enumerate(names):
        mk = (tl >= bins[i]) & (tl < bins[i + 1])
        if mk.sum() > 0:
            rows.append(
                {
                    "Faixa": nm,
                    "Tokens": f"{int(bins[i])}-{int(bins[i+1]-1)}",
                    "N": int(mk.sum()),
                    "Acc": accuracy_score(y_test[mk], predictions[mk]),
                    "F1": f1_score(
                        y_test[mk], predictions[mk], average="macro", zero_division=0
                    ),
                }
            )
    df = pd.DataFrame(rows).round(4)
    if print_table:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


def error_distribution_full(
    df_test_meta: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None,
    model_name: str = "Model",
    save_as_prefix: str = "error_distrib",
) -> dict[str, pd.DataFrame]:
    """Distribuicao de erros por fonte, categoria, ano e confianca.

    Args:
        df_test_meta: DataFrame de teste com colunas opcionais
            (source, category, date_parsed).
        y_test, y_pred: ndarrays (mesmo tamanho).
        probs: matriz de probabilidades (N, C) para coluna de confianca.
            Se None, omite analise por confianca.
        save_as_prefix: prefixo para os CSVs gerados.

    Returns:
        dict {dimensao: DataFrame} para cada analise gerada.
    """
    if len(y_test) != len(df_test_meta):
        log.warning(
            f"Tamanho df_test_meta ({len(df_test_meta)}) != y_test ({len(y_test)})."
            " Pulando error_distribution_full."
        )
        return {}

    y_test = np.asarray(y_test).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    correct = (y_test == y_pred)
    out: dict[str, pd.DataFrame] = {}

    def _by(col: str, name: str) -> pd.DataFrame | None:
        if col not in df_test_meta.columns:
            return None
        rows = []
        for val in df_test_meta[col].dropna().unique():
            m = (df_test_meta[col] == val).to_numpy()
            if m.sum() < 10:
                continue
            n_err = int(((~correct) & m).sum())
            n_tot = int(m.sum())
            err_rate = n_err / n_tot
            rows.append({
                name: val,
                "N": n_tot,
                "Erros": n_err,
                "Taxa erro": round(err_rate, 4),
                "Acc": round(1 - err_rate, 4),
                "F1": round(f1_score(
                    y_test[m], y_pred[m], average="macro", zero_division=0
                ), 4),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("Taxa erro", ascending=False)
        save_table(df, f"{save_as_prefix}_by_{name.lower()}")
        return df

    by_source = _by("source", "Fonte")
    if by_source is not None:
        out["source"] = by_source

    by_category = _by("category", "Categoria")
    if by_category is not None:
        out["category"] = by_category

    # Por ano (extraido de date_parsed)
    if "date_parsed" in df_test_meta.columns:
        years = pd.to_datetime(df_test_meta["date_parsed"], errors="coerce").dt.year
        if years.notna().sum() > 0:
            tmp = df_test_meta.copy()
            tmp["_year"] = years
            rows = []
            for yr in sorted(tmp["_year"].dropna().unique()):
                m = (tmp["_year"] == yr).to_numpy()
                if m.sum() < 10:
                    continue
                n_err = int(((~correct) & m).sum())
                n_tot = int(m.sum())
                err_rate = n_err / n_tot
                rows.append({
                    "Ano": int(yr),
                    "N": n_tot,
                    "Erros": n_err,
                    "Taxa erro": round(err_rate, 4),
                    "Acc": round(1 - err_rate, 4),
                    "F1": round(f1_score(
                        y_test[m], y_pred[m], average="macro", zero_division=0
                    ), 4),
                })
            if rows:
                df_year = pd.DataFrame(rows).sort_values("Ano")
                save_table(df_year, f"{save_as_prefix}_by_year")
                out["year"] = df_year

    # Por confianca (quartis de max(probs))
    if probs is not None and probs.ndim == 2:
        conf = probs.max(axis=1)
        qs = np.percentile(conf, [25, 50, 75])
        bins = [0, qs[0], qs[1], qs[2], 1.0001]
        names = ["Q1 (baixa)", "Q2", "Q3", "Q4 (alta)"]
        rows = []
        for i, nm in enumerate(names):
            m = (conf >= bins[i]) & (conf < bins[i + 1])
            if m.sum() == 0:
                continue
            n_err = int(((~correct) & m).sum())
            n_tot = int(m.sum())
            err_rate = n_err / n_tot
            rows.append({
                "Quartil_conf": nm,
                "Faixa": f"{bins[i]:.3f}-{bins[i+1]:.3f}",
                "N": n_tot,
                "Erros": n_err,
                "Taxa erro": round(err_rate, 4),
                "Acc": round(1 - err_rate, 4),
            })
        if rows:
            df_conf = pd.DataFrame(rows)
            save_table(df_conf, f"{save_as_prefix}_by_confidence")
            out["confidence"] = df_conf

    log.info(
        f"[Error analysis] {model_name}: dimensoes geradas={list(out.keys())}"
    )
    return out


__all__ = [
    "error_summary",
    "performance_by_length",
    "error_distribution_full",
]
