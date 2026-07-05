"""Carregamento do FakeRecogna 2.0, integridade, splits e Fake.br-Corpus.

Submódulos:
- `loading`: Carregamento HF + normalização de schema + dedupe.
- `integrity_checks`: Fonte/comprimento/temporal/NER por classe.
- `splits`: Splits random/temporal/source.
- `fakebr_loader`: Loader local do Fake.br-Corpus.
- `pipeline`: Orquestrador `run(ctx)` que executa o subpacote inteiro.
"""

from . import anti_bias_splits, fakebr_loader, integrity_checks, loading, pipeline, splits
from .anti_bias_splits import make_anti_bias_splits
from .fakebr_loader import load_fakebr_corpus
from .integrity_checks import (
    length_by_class,
    ner_top_by_class,
    source_class_analysis,
    temporal_analysis,
)
from .loading import (
    dedupe,
    encode_labels,
    load_and_prepare,
    load_fakerecogna,
    normalize_schema,
    parse_dates,
)
from .pipeline import run as run_pipeline
from .splits import make_random_splits, make_source_splits, make_temporal_splits

__all__ = [
    # submódulos
    "loading",
    "integrity_checks",
    "splits",
    "fakebr_loader",
    "pipeline",
    # loading
    "load_fakerecogna",
    "normalize_schema",
    "encode_labels",
    "parse_dates",
    "dedupe",
    "load_and_prepare",
    # integrity
    "source_class_analysis",
    "length_by_class",
    "temporal_analysis",
    "ner_top_by_class",
    # splits
    "make_random_splits",
    "make_temporal_splits",
    "make_source_splits",
    "make_anti_bias_splits",
    "anti_bias_splits",
    # fakebr
    "load_fakebr_corpus",
    # pipeline
    "run_pipeline",
]
