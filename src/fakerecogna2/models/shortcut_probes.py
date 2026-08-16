"""Sondas de atalho: classificadores triviais que quantificam sinal superficial.

Motivação (Revisor 1, ENIAC 2026): a forma mais direta de expor um atalho é
mostrar que um classificador *trivial*, sem acesso ao conteúdo semântico,
recupera grande parte do desempenho. Três famílias de sondas, todas
Regressão Logística sobre o MESMO split oficial (70/10/20, SEED):

- ``length``: apenas o número de tokens do texto pré-processado (1 feature).
  Mede o atalho da assimetria de sumarização (só a classe verdadeira é
  sumarizada).
- ``top-K termos``: presença binária dos K termos mais discriminativos por
  classe segundo log-odds com prior de Dirichlet, calculados **apenas no
  treino** (2K features).
- ``metadados``: apenas categoria (one-hot das 10 mais frequentes + outros)
  e ano de publicação — nenhum acesso ao texto.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .._context import ExperimentContext
from ..config import SEED
from ..statistics.log_odds import build_ngram_counts, log_odds_dirichlet
from ..utils.io_utils import save_table
from ..utils.logging_utils import get_logger

log = get_logger()


def _official_split_indices(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Réplica exata da indexação de `data.splits.make_random_splits`.

    Mesmos parâmetros (70/10/20, estratificado, mesma seed) ⇒ mesmos índices
    do split oficial. Necessária porque o pipeline não preserva os índices de
    treino no ctx (apenas `df_test_meta`).
    """
    idx = np.arange(len(df))
    y = df["label_enc"].to_numpy()
    idx_tv, idx_te, y_tv, _ = train_test_split(
        idx, y, test_size=0.20, stratify=y, random_state=seed
    )
    idx_tr, idx_vl, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.125, stratify=y_tv, random_state=seed
    )
    return idx_tr, idx_vl, idx_te


def _fit_eval(
    name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_features: int,
    seed: int = SEED,
) -> dict:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    row = {
        "probe": name,
        "n_features": int(n_features),
        "Accuracy": round(float(accuracy_score(y_te, yp)), 4),
        "F1": round(float(f1_score(y_te, yp, average="macro")), 4),
    }
    log.info(f'[probe] {name}: Acc={row["Accuracy"]} F1={row["F1"]}')
    return row


def run_shortcut_probes(
    ctx: ExperimentContext,
    top_ks: tuple[int, ...] = (6, 20, 50),
    save_as: str | None = "23_shortcut_probes",
) -> pd.DataFrame:
    """Roda as sondas de atalho no split oficial e salva a tabela.

    Requer `ctx` após `preprocessing.run_pipeline` (campos `X_*_text`,
    `y_*`, `df`). Retorna DataFrame [probe, n_features, Accuracy, F1].
    """
    X_tr_text, X_te_text = ctx.X_train_text, ctx.X_test_text
    y_tr, y_te = np.asarray(ctx.y_train), np.asarray(ctx.y_test)
    rows: list[dict] = []

    # -- 1) Comprimento apenas -------------------------------------------------
    len_tr = np.array([[len(t.split())] for t in X_tr_text], dtype=float)
    len_te = np.array([[len(t.split())] for t in X_te_text], dtype=float)
    scaler = StandardScaler().fit(len_tr)
    rows.append(_fit_eval(
        "Nº de tokens (1 feature)",
        scaler.transform(len_tr), y_tr, scaler.transform(len_te), y_te,
        n_features=1, seed=ctx.seed,
    ))

    # -- 2) Top-K termos por log-odds (calculados só no treino) ---------------
    tr_by_class: dict[int, list[str]] = {
        0: [t for t, y in zip(X_tr_text, y_tr) if y == 0],
        1: [t for t, y in zip(X_tr_text, y_tr) if y == 1],
    }
    counts_a, _ = build_ngram_counts(tr_by_class[0], n=1)
    counts_b, _ = build_ngram_counts(tr_by_class[1], n=1)
    lo = log_odds_dirichlet(Counter(counts_a), Counter(counts_b))
    for k in top_ks:
        vocab = pd.concat([lo.head(k), lo.tail(k)])["term"].tolist()
        cv = CountVectorizer(vocabulary=vocab, binary=True)
        rows.append(_fit_eval(
            f"Top-{k} termos por classe (log-odds, {2 * k} features)",
            cv.transform(X_tr_text), y_tr, cv.transform(X_te_text), y_te,
            n_features=2 * k, seed=ctx.seed,
        ))

    # -- 3) Metadados apenas (categoria + ano; sem texto) ----------------------
    df = ctx.df
    if df is not None and "category" in df.columns:
        idx_tr, _, idx_te = _official_split_indices(df, seed=ctx.seed)
        top_cats = df["category"].fillna("outros").value_counts().head(10).index
        cat = df["category"].fillna("outros").where(
            df["category"].fillna("outros").isin(top_cats), "outros"
        )
        meta = pd.get_dummies(cat, prefix="cat")
        if "date_parsed" in df.columns:
            year = df["date_parsed"].dt.year
            meta["year"] = year.fillna(0).astype(float)
            meta["year_missing"] = year.isna().astype(float)
        M = meta.to_numpy(dtype=float)
        M_tr, M_te = M[idx_tr], M[idx_te]
        sc = StandardScaler().fit(M_tr)
        rows.append(_fit_eval(
            f"Metadados apenas (categoria+ano, {M.shape[1]} features)",
            sc.transform(M_tr), df["label_enc"].to_numpy()[idx_tr],
            sc.transform(M_te), df["label_enc"].to_numpy()[idx_te],
            n_features=M.shape[1], seed=ctx.seed,
        ))
    else:
        log.warning("[probe] coluna `category` ausente — sonda de metadados pulada.")

    out = pd.DataFrame(rows)
    if save_as is not None:
        save_table(out, save_as)
    return out


def class_support_by_year(
    ctx: ExperimentContext,
    save_as: str | None = "23_test_support_by_year",
) -> pd.DataFrame | None:
    """Suporte por classe × ano no conjunto de teste oficial.

    Base factual para a leitura da tabela de F1 por ano do artigo (o colapso
    de F1-macro em anos com uma classe quase ausente é mecânico). Usa
    `ctx.extras['df_test_meta']` (metadados preservados do split).
    """
    dtm = ctx.extras.get("df_test_meta")
    if dtm is None or "date_parsed" not in dtm.columns:
        log.warning("[probe] df_test_meta indisponível — suporte por ano pulado.")
        return None
    d = dtm.dropna(subset=["date_parsed"]).copy()
    d["ano"] = d["date_parsed"].dt.year.astype(int)
    tab = (
        d.groupby(["ano", "label_enc"]).size().unstack(fill_value=0)
        .rename(columns={0: "N_verdadeira", 1: "N_fake"})
        .reset_index()
    )
    tab["N"] = tab["N_verdadeira"] + tab["N_fake"]
    if save_as is not None:
        save_table(tab, save_as)
    log.info(f"[probe] suporte por classe×ano:\n{tab.to_string(index=False)}")
    return tab


__all__ = ["run_shortcut_probes", "class_support_by_year"]
