"""Vetorização TF-IDF pra baselines clássicos (parte da cell 27)."""

from __future__ import annotations


from sklearn.feature_extraction.text import TfidfVectorizer




# -- API limpa ----------------------------------------------------------------
def fit_tfidf(
    X_train: list[str],
    max_features: int = 30000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    stop_words: list[str] | None = None,
) -> TfidfVectorizer:
    """Fita um TfidfVectorizer com os defaults do experimento."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )
    tfidf.fit(X_train)
    return tfidf


__all__ = [
    "fit_tfidf",
]
