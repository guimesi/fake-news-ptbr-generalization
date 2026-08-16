"""log-odds com Dirichlet prior (Monroe, Colaresi & Quinn 2008) + Chi²/TF-IDF.

Cobre as células 15–18 do notebook (Seção 4): análise lexical das classes.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

from ..preprocessing.text_cleaning import get_stopwords_pt
from ..utils.io_utils import save_table




# -- API limpa ----------------------------------------------------------------
def log_odds_dirichlet(
    counts_a: Counter,
    counts_b: Counter,
    prior: dict | None = None,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Log-odds-ratio com prior Dirichlet (Monroe et al. 2008).

    Args:
        counts_a/counts_b: Counters de termos por classe.
        prior: dict termo→pseudo-count, ou None pra prior uniforme `alpha`.
        alpha: valor do prior uniforme.

    Returns:
        DataFrame ordenado por z descendente: term, count_a, count_b, log_odds, z.
        z > 1.96 → significativo para A; z < -1.96 → significativo para B.
    """
    vocab = set(counts_a) | set(counts_b)
    if prior is None:
        prior = {w: alpha for w in vocab}
    na = sum(counts_a.values())
    nb = sum(counts_b.values())
    n0 = sum(prior.values())
    rows = []
    for w in vocab:
        ya, yb, y0 = counts_a.get(w, 0), counts_b.get(w, 0), prior.get(w, alpha)
        lo = math.log((ya + y0) / (na + n0 - ya - y0)) - math.log(
            (yb + y0) / (nb + n0 - yb - y0)
        )
        var = 1 / (ya + y0) + 1 / (yb + y0)
        z = lo / math.sqrt(var)
        rows.append((w, ya, yb, lo, z))
    return pd.DataFrame(
        rows, columns=["term", "count_a", "count_b", "log_odds", "z"]
    ).sort_values("z", ascending=False)


def build_ngram_counts(
    texts, n: int = 1, max_vocab: int = 50000
) -> tuple[dict[str, int], CountVectorizer]:
    """Conta n-gramas removendo stopwords PT. Retorna (dict, vectorizer)."""
    cv = CountVectorizer(
        ngram_range=(n, n),
        max_features=max_vocab,
        token_pattern=r"(?u)\b\w[\w-]+\b",
        stop_words=list(get_stopwords_pt()),
    )
    X = cv.fit_transform(texts)
    vocab = cv.get_feature_names_out()
    return dict(zip(vocab, np.asarray(X.sum(0)).flatten())), cv


def lex_analysis(
    df: pd.DataFrame,
    name: str = "abstrativa",
    top_k: int = 25,
    save: bool = True,
) -> dict[int, pd.DataFrame] | None:
    """Roda log-odds em uni/bi/tri-gramas pra cada classe (binário só)."""
    class_names = sorted(df["label"].unique())
    if len(class_names) != 2:
        print("log-odds implementado pra binário; pulando.")
        return None

    results: dict[int, pd.DataFrame] = {}
    for ngram in [1, 2, 3]:
        cls_a, cls_b = class_names
        ca, _ = build_ngram_counts(df[df["label"] == cls_a]["text"].astype(str), n=ngram)
        cb, _ = build_ngram_counts(df[df["label"] == cls_b]["text"].astype(str), n=ngram)
        lo = log_odds_dirichlet(Counter(ca), Counter(cb))
        results[ngram] = lo
        print(f"\n=== {name} — n={ngram} ===")
        print(f'Top {top_k} associados a "{cls_a}":')
        print(lo.head(top_k)[["term", "count_a", "count_b", "z"]].to_string(index=False))
        print(f'\nTop {top_k} associados a "{cls_b}":')
        print(
            lo.tail(top_k)
            .iloc[::-1][["term", "count_a", "count_b", "z"]]
            .to_string(index=False)
        )
        if save:
            save_table(lo.head(100), f"04_logodds_{name}_n{ngram}_top_{cls_a}")
            save_table(lo.tail(100).iloc[::-1], f"04_logodds_{name}_n{ngram}_top_{cls_b}")
    return results


def chi2_top(
    df: pd.DataFrame,
    class_names: list[str],
    k: int = 30,
    save: bool = True,
) -> pd.DataFrame:
    """Chi² top features (TF-IDF como vetorização). Retorna DataFrame top-300."""
    tf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=list(get_stopwords_pt()),
    )
    X = tf.fit_transform(df["text"].astype(str))
    y = df["label_enc"].values
    chi, p = chi2(X, y)
    vocab = tf.get_feature_names_out()
    df_chi = pd.DataFrame({"term": vocab, "chi2": chi, "p": p}).sort_values(
        "chi2", ascending=False
    )
    num_classes = len(class_names)

    vocab_list = vocab.tolist()

    def classify_side(term: str) -> str:
        col = vocab_list.index(term)
        vec = X[:, col].toarray().flatten()
        m = [vec[y == c].mean() for c in range(num_classes)]
        return class_names[int(np.argmax(m))]

    df_chi["characteristic_of"] = df_chi.head(500)["term"].apply(classify_side)
    print(f"\n=== Chi² top-{k} termos ===")
    print(df_chi.head(k).to_string(index=False))
    if save:
        save_table(df_chi.head(300), "04_chi2_top_abstrativa")
    return df_chi


def show_examples_for_term(
    df: pd.DataFrame,
    term: str,
    cls: str,
    n: int = 3,
    max_chars: int = 300,
) -> None:
    """Imprime n exemplos onde `term` aparece em textos da classe `cls`."""
    mask = (
        df["text"]
        .astype(str)
        .str.lower()
        .str.contains(rf"\b{re.escape(term)}\b", regex=True)
    )
    sub = df[mask & (df["label"] == cls)].head(n)
    print(f'\n--- Exemplos: "{term}" em classe "{cls}" ---')
    for _, row in sub.iterrows():
        snippet = str(row["text"])[:max_chars].replace("\n", " ")
        print(f"  • {snippet}…")


__all__ = [
    "log_odds_dirichlet",
    "build_ngram_counts",
    "lex_analysis",
    "chi2_top",
    "show_examples_for_term",
]
