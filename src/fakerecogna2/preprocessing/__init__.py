"""Limpeza de texto, sumarização extrativa (TextRank) e equalizador linguístico.

Submódulos:
- `text_cleaning`: preprocess_base / _stem / _lemma + recursos NLP lazy.
- `paraphrasing`: TextRank + build_equalized_dataset.
- `equalizer`: METATEXT_FAKECHECK + strip_metatext + build_nometatext_dataset.
- `pipeline`: orquestrador `run(ctx)`.
"""

from . import equalizer, paraphrasing, pipeline, text_cleaning
from .equalizer import METATEXT_FAKECHECK, build_nometatext_dataset, strip_metatext
from .paraphrasing import (
    build_equalized_dataset,
    estimate_n_sentences,
    extractive_summarize,
)
from .pipeline import run as run_pipeline
from .text_cleaning import (
    PREP_FNS,
    apply_preprocessing,
    get_spacy_pt,
    get_stemmer_pt,
    get_stopwords_pt,
    preprocess_base,
    preprocess_lemma,
    preprocess_stem,
)

__all__ = [
    "text_cleaning",
    "paraphrasing",
    "equalizer",
    "pipeline",
    "PREP_FNS",
    "preprocess_base",
    "preprocess_stem",
    "preprocess_lemma",
    "apply_preprocessing",
    "get_stopwords_pt",
    "get_stemmer_pt",
    "get_spacy_pt",
    "extractive_summarize",
    "estimate_n_sentences",
    "build_equalized_dataset",
    "METATEXT_FAKECHECK",
    "strip_metatext",
    "build_nometatext_dataset",
    "run_pipeline",
]
