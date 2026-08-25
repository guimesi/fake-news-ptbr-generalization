"""Integracao XAI <-> analise de erros.

Cruza explicabilidade local (LIME por exemplo, agregada por TP/TN/FP/FN) com
estratificacao de erros por dimensao (fonte, categoria, ano, confianca).

Cobre o item 60 do STATUS.md: 'integracao XAI <-> erros consolidada'.

Artefatos gerados:
  - `xai_err_cell_per_example.csv`: tabela por exemplo (idx, true, pred, cell,
    source, category, year, confidence).
  - `xai_err_cell_by_<dim>.csv`: contagem de TP/TN/FP/FN por valor de cada
    dimensao (categoria, fonte, ano).
  - `xai_err_top_tokens_by_dim_cell.csv`: para combinacoes (categoria, celula),
    quais tokens LIME aparecem com mais peso (se LIME tiver sido executado
    em exemplos desta combinacao).
  - `xai_err_consolidated.md`: relatorio narrativo apontando categorias com
    erro acima da media e os tokens LIME mais influentes naquela faixa.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..explainability.lime_explainer import _parse_lime_features
from ..utils.io_utils import save_table
from ..utils.logging_utils import get_logger

log = get_logger()


def build_cell_per_example(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class_idx: int = 0,
) -> np.ndarray:
    """Retorna ndarray de strings 'TP'/'TN'/'FP'/'FN' por exemplo.

    Convencao canonica do corpus: 0 = verdadeira, 1 = fake (CHANGELOG §14);
    `positive_class_idx` define a classe tabulada como positiva (default 0).
    """
    pos = positive_class_idx
    neg = 1 - pos
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cells = np.full(len(y_true), "??", dtype=object)
    cells[(y_true == pos) & (y_pred == pos)] = "TP"
    cells[(y_true == neg) & (y_pred == neg)] = "TN"
    cells[(y_true == neg) & (y_pred == pos)] = "FP"
    cells[(y_true == pos) & (y_pred == neg)] = "FN"
    return cells


def cross_cell_by_dimension(
    cells: np.ndarray,
    df_meta: pd.DataFrame,
    dim_col: str,
    dim_name: str | None = None,
    min_count: int = 20,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Cruza celulas (TP/TN/FP/FN) por valor de uma dimensao.

    Retorna DataFrame com colunas: dim, N, TP, TN, FP, FN, erro_rate.
    Filtra valores com menos de `min_count` exemplos.
    """
    if dim_col not in df_meta.columns:
        log.warning(f"[xai-err] dim_col '{dim_col}' ausente em df_meta")
        return pd.DataFrame()

    dim_name = dim_name or dim_col
    rows = []
    for val in df_meta[dim_col].dropna().unique():
        m = (df_meta[dim_col] == val).to_numpy()
        n = int(m.sum())
        if n < min_count:
            continue
        c = cells[m]
        n_tp = int((c == "TP").sum())
        n_tn = int((c == "TN").sum())
        n_fp = int((c == "FP").sum())
        n_fn = int((c == "FN").sum())
        n_err = n_fp + n_fn
        rows.append({
            dim_name: val,
            "N": n,
            "TP": n_tp,
            "TN": n_tn,
            "FP": n_fp,
            "FN": n_fn,
            "Taxa erro": round(n_err / max(1, n), 4),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Taxa erro", ascending=False)
    if save_as:
        save_table(df, save_as)
    return df


def aggregate_lime_tokens_by_dim_cell(
    lime_records_df: pd.DataFrame,
    cells: np.ndarray,
    df_meta: pd.DataFrame,
    dim_col: str,
    top_n: int = 10,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Agrega tokens LIME por (dim, celula).

    `lime_records_df` deve ter colunas idx, tag, features (str do as_list LIME).
    `cells` indexado por posicao no test set.
    `df_meta` com a dimensao.

    Para cada (dim_value, cell), soma pesos absolutos por token e retorna
    top_n.
    """
    if "idx" not in lime_records_df.columns or "features" not in lime_records_df.columns:
        log.warning("[xai-err] lime_records_df sem colunas 'idx'/'features'")
        return pd.DataFrame()
    if dim_col not in df_meta.columns:
        return pd.DataFrame()

    # idx -> features parsed
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    n_examples: dict[tuple[str, str], int] = defaultdict(int)

    for _, row in lime_records_df.iterrows():
        idx = int(row["idx"])
        if idx >= len(cells) or idx >= len(df_meta):
            continue
        cell = cells[idx]
        dim_val = df_meta.iloc[idx].get(dim_col)
        if pd.isna(dim_val):
            continue
        key = (str(dim_val), str(cell))
        n_examples[key] += 1
        feats = _parse_lime_features(str(row["features"]))
        if not feats:
            continue
        for tok, w in feats:
            agg[key][tok] += abs(float(w))

    rows = []
    for (dim_val, cell), tok_w in agg.items():
        top = sorted(tok_w.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for rank, (tok, w) in enumerate(top, 1):
            rows.append({
                dim_col: dim_val,
                "cell": cell,
                "rank": rank,
                "token": tok,
                "abs_weight": round(w, 4),
                "n_examples": n_examples[(dim_val, cell)],
            })

    df = pd.DataFrame(rows)
    if save_as and not df.empty:
        save_table(df, save_as)
    return df


def consolidate_xai_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None,
    df_meta: pd.DataFrame,
    lime_records_df: pd.DataFrame | None = None,
    model_name: str = "Ens3",
    positive_class_idx: int = 0,
    save_as_prefix: str = "xai_err",
) -> dict[str, pd.DataFrame]:
    """Pipeline completo de integracao XAI <-> erros.

    Returns: dict com dataframes gerados.
    """
    cells = build_cell_per_example(y_true, y_pred, positive_class_idx)

    out: dict[str, pd.DataFrame] = {}

    # 1) Tabela exemplo por exemplo
    per_example = pd.DataFrame({
        "idx": np.arange(len(cells)),
        "true": y_true.astype(int),
        "pred": np.asarray(y_pred).astype(int),
        "cell": cells,
    })
    if probs is not None and probs.ndim == 2:
        per_example["confidence"] = probs.max(axis=1).round(4)
    for col in ("source", "category", "date_parsed"):
        if col in df_meta.columns:
            per_example[col] = df_meta[col].values
    save_table(per_example, f"{save_as_prefix}_cell_per_example")
    out["per_example"] = per_example

    # 2) Cross-tabs por dimensao
    for dim_col, dim_name in [
        ("source", "Fonte"),
        ("category", "Categoria"),
    ]:
        df_dim = cross_cell_by_dimension(
            cells, df_meta, dim_col, dim_name=dim_name,
            save_as=f"{save_as_prefix}_cell_by_{dim_name.lower()}",
        )
        if not df_dim.empty:
            out[dim_col] = df_dim

    # Por ano (extraido de date_parsed)
    if "date_parsed" in df_meta.columns:
        years = pd.to_datetime(df_meta["date_parsed"], errors="coerce").dt.year
        if years.notna().sum() > 0:
            tmp = df_meta.copy()
            tmp["_year"] = years
            df_year = cross_cell_by_dimension(
                cells, tmp, "_year", dim_name="Ano",
                save_as=f"{save_as_prefix}_cell_by_ano",
            )
            if not df_year.empty:
                out["year"] = df_year

    # 3) Tokens LIME por (categoria, celula)
    if lime_records_df is not None and not lime_records_df.empty:
        df_tok_cat = aggregate_lime_tokens_by_dim_cell(
            lime_records_df, cells, df_meta, dim_col="category", top_n=10,
            save_as=f"{save_as_prefix}_top_tokens_by_category_cell",
        )
        if not df_tok_cat.empty:
            out["tokens_by_category_cell"] = df_tok_cat

    # 4) Relatorio narrativo markdown
    md_lines = [
        f"# Integracao XAI <-> Erros - {model_name}",
        "",
        f"Total de exemplos: {len(cells)}",
        f"- TP: {(cells == 'TP').sum()}",
        f"- TN: {(cells == 'TN').sum()}",
        f"- FP: {(cells == 'FP').sum()}",
        f"- FN: {(cells == 'FN').sum()}",
        "",
        "## Categorias com maior taxa de erro",
        "",
    ]
    if "category" in out:
        df_cat = out["category"].head(10)
        md_lines.append(df_cat.to_markdown(index=False))
        md_lines.append("")
    if "tokens_by_category_cell" in out:
        md_lines.append("## Tokens LIME por categoria x celula (top 5)")
        md_lines.append("")
        df_tok = out["tokens_by_category_cell"]
        for (cat, cell), sub in df_tok.groupby(["category", "cell"]):
            top5 = sub.head(5)["token"].tolist()
            md_lines.append(f"- **{cat} / {cell}**: " + ", ".join(top5))
        md_lines.append("")

    md_path = Path("outputs/metrics") / f"{save_as_prefix}_consolidated.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info(f"[xai-err] relatorio markdown salvo: {md_path.name}")

    return out


__all__ = [
    "build_cell_per_example",
    "cross_cell_by_dimension",
    "aggregate_lime_tokens_by_dim_cell",
    "consolidate_xai_errors",
]
