"""Pipelines de pré-processamento de texto.

Expõe `preprocess_base` e `preprocess_base_A`.
"""

from __future__ import annotations

import re
from typing import Callable

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

URL_RE = re.compile(r"http\S+|www\.\S+")
MAIL_RE = re.compile(r"\S+@\S+|@\w+")
NONALPHA_RE = re.compile(r"[^a-záàâãéèêíïóôõúüç\s]")




# -- Recursos NLP (lazy-load) -------------------------------------------------
_STOPWORDS_PT: set[str] | None = None
_STEMMER_PT: RSLPStemmer | None = None
_NLP_PT = None


def _ensure_nltk_resources() -> None:
    for resource, downloader_name in [
        ("corpora/stopwords", "stopwords"),
        ("stemmers/rslp", "rslp"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(downloader_name, quiet=True)


def get_stopwords_pt() -> set[str]:
    """Lista de stopwords em PT (NLTK), com lazy-load."""
    global _STOPWORDS_PT
    if _STOPWORDS_PT is None:
        _ensure_nltk_resources()
        _STOPWORDS_PT = set(stopwords.words("portuguese"))
    return _STOPWORDS_PT


def get_stemmer_pt() -> RSLPStemmer:
    """Stemmer RSLP, com lazy-load."""
    global _STEMMER_PT
    if _STEMMER_PT is None:
        _ensure_nltk_resources()
        _STEMMER_PT = RSLPStemmer()
    return _STEMMER_PT


def get_spacy_pt():
    """spaCy `pt_core_news_sm` (sem parser/ner), com lazy-load."""
    global _NLP_PT
    if _NLP_PT is None:
        import spacy
        _NLP_PT = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    return _NLP_PT


# -- Funções de pré-processamento ---------------------------------------------
def preprocess_base(text: str) -> str:
    """Lowercase + remoção URLs/emails/não-alfa + stopwords + filtro len>2."""
    stopwords_pt = get_stopwords_pt()
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = MAIL_RE.sub(" ", t)
    t = NONALPHA_RE.sub(" ", t)
    return " ".join(w for w in t.split() if w not in stopwords_pt and len(w) > 2)


def preprocess_stem(text: str) -> str:
    """Base + stemming RSLP."""
    stemmer = get_stemmer_pt()
    return " ".join(stemmer.stem(w) for w in preprocess_base(text).split())


def preprocess_lemma(text: str) -> str:
    """Base + lematização spaCy."""
    nlp = get_spacy_pt()
    doc = nlp(preprocess_base(text))
    return " ".join(t.lemma_ for t in doc if len(t.lemma_) > 2)


PREP_FNS: dict[str, Callable[[str], str]] = {
    "base": preprocess_base,
    "stem": preprocess_stem,
    "lemma": preprocess_lemma,
}


def apply_preprocessing(
    df,
    text_col: str = "text",
    out_col: str = "text_proc",
    method: str = "base",
    min_length: int = 10,
):
    """Aplica `method` em `df[text_col]`, escreve em `df[out_col]`, filtra textos curtos."""
    from tqdm.auto import tqdm

    if method not in PREP_FNS:
        raise ValueError(f"method={method!r} inválido. Opções: {list(PREP_FNS)}")
    fn = PREP_FNS[method]
    tqdm.pandas(desc=f"Preprocess {method}")
    out = df.copy()
    out[out_col] = out[text_col].progress_apply(fn)
    return out[out[out_col].str.len() > min_length].reset_index(drop=True)


__all__ = [
    "URL_RE",
    "MAIL_RE",
    "NONALPHA_RE",
    "PREP_FNS",
    "get_stopwords_pt",
    "get_stemmer_pt",
    "get_spacy_pt",
    "preprocess_base",
    "preprocess_stem",
    "preprocess_lemma",
    "apply_preprocessing",
]
