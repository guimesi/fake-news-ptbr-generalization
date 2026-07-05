"""Embeddings BERTimbau, DataLoaders, TF-IDF, streaming."""

from . import dataloaders, embeddings, streaming, vectorization
from .dataloaders import make_loaders
from .embeddings import EmbeddingExtractor, extract_all_embeddings
from .streaming import (
    StreamingEmbDataset,
    free_embeddings,
    make_streaming_collate,
    make_streaming_loaders,
)
from .vectorization import fit_tfidf

__all__ = [
    "embeddings",
    "dataloaders",
    "vectorization",
    "streaming",
    "EmbeddingExtractor",
    "extract_all_embeddings",
    "make_loaders",
    "fit_tfidf",
    "StreamingEmbDataset",
    "make_streaming_collate",
    "make_streaming_loaders",
    "free_embeddings",
]
