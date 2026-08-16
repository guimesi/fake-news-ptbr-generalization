"""Testes pra API do subpacote `fakerecogna2.data`.

Cobrem loading/splits/integrity sem baixar o dataset real — usam DataFrames
sintéticos pequenos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fakerecogna2 import ExperimentContext
from fakerecogna2.data import (
    dedupe,
    encode_labels,
    length_by_class,
    make_random_splits,
    make_source_splits,
    make_temporal_splits,
    normalize_schema,
    parse_dates,
    source_class_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synth_df() -> pd.DataFrame:
    """DataFrame sintético com a forma esperada do FakeRecogna."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "Title": [f"Título {i}" for i in range(n)],
            "News": [f"Notícia número {i} com texto razoável." for i in range(n)],
            "URL": [
                f"https://www.fonte{i % 5}.com/path/{i}" for i in range(n)
            ],
            "Date": [f"{(i % 28) + 1:02d}/03/2024" for i in range(n)],
            "Label": rng.choice(["fake", "real"], size=n).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# normalize_schema
# ---------------------------------------------------------------------------
def test_normalize_schema_renames_columns(synth_df):
    out = normalize_schema(synth_df)
    assert "text" in out.columns
    assert "title" in out.columns
    assert "url" in out.columns
    assert "date" in out.columns
    assert "label" in out.columns


def test_normalize_schema_derives_source(synth_df):
    out = normalize_schema(synth_df)
    assert "source" in out.columns
    # cada URL deve dar o domínio
    assert all(s.startswith("fonte") for s in out["source"])


def test_normalize_schema_preserves_size(synth_df):
    assert len(normalize_schema(synth_df)) == len(synth_df)


# ---------------------------------------------------------------------------
# encode_labels
# ---------------------------------------------------------------------------
def test_encode_labels_creates_label_enc(synth_df):
    df = normalize_schema(synth_df)
    out, enc, names = encode_labels(df)
    assert "label_enc" in out.columns
    assert out["label_enc"].dtype.kind in ("i", "u")
    assert set(names) == {"fake", "real"}


def test_encode_labels_reuses_encoder(synth_df):
    df = normalize_schema(synth_df)
    out1, enc, _ = encode_labels(df)
    out2, enc2, _ = encode_labels(df, encoder=enc)
    assert enc is enc2
    np.testing.assert_array_equal(out1["label_enc"], out2["label_enc"])


# ---------------------------------------------------------------------------
# parse_dates
# ---------------------------------------------------------------------------
def test_parse_dates_creates_date_parsed(synth_df):
    df = normalize_schema(synth_df)
    out = parse_dates(df)
    assert "date_parsed" in out.columns
    # Maioria das datas válidas
    assert out["date_parsed"].notna().sum() > len(out) * 0.9


def test_parse_dates_absent_column():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["fake", "real"]})
    out = parse_dates(df)
    assert "date_parsed" in out.columns
    assert out["date_parsed"].isna().all()


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------
def test_dedupe_removes_exact_duplicates():
    df = pd.DataFrame({"text": ["a b c", "a b c", "x y z"]})
    out, n_ex, n_nr = dedupe(df, text_col="text")
    assert n_ex == 1
    assert len(out) == 2


def test_dedupe_preserves_unique():
    # Textos completamente distintos (sem shingles compartilhados em comum)
    df = pd.DataFrame(
        {
            "text": [
                "matemática algébrica polinomial",
                "futebol brasileiro campeonato 2024",
                "receita de bolo chocolate cobertura",
                "geografia montanhas andinas peru",
                "música clássica violino bach concerto",
            ]
        }
    )
    out, n_ex, _ = dedupe(df, text_col="text")
    assert n_ex == 0
    assert len(out) == 5


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_random_splits_proportions(synth_df):
    df = normalize_schema(synth_df)
    df, _, _ = encode_labels(df)
    X_tr, X_vl, X_te, y_tr, y_vl, y_te = make_random_splits(
        df, text_col="text", seed=42
    )
    total = len(X_tr) + len(X_vl) + len(X_te)
    assert total == len(df)
    # ~70/10/20
    assert 0.65 * total <= len(X_tr) <= 0.75 * total
    assert 0.15 * total <= len(X_te) <= 0.25 * total


def test_random_splits_are_stratified(synth_df):
    df = normalize_schema(synth_df)
    df, _, _ = encode_labels(df)
    _, _, _, y_tr, y_vl, y_te = make_random_splits(df, text_col="text", seed=42)
    # Proporção de classe ~similar em cada split
    p_tr = np.mean(y_tr)
    p_te = np.mean(y_te)
    assert abs(p_tr - p_te) < 0.15


def test_random_splits_reproducible(synth_df):
    df = normalize_schema(synth_df)
    df, _, _ = encode_labels(df)
    out_a = make_random_splits(df, text_col="text", seed=42)
    out_b = make_random_splits(df, text_col="text", seed=42)
    np.testing.assert_array_equal(out_a[3], out_b[3])  # y_train
    np.testing.assert_array_equal(out_a[5], out_b[5])  # y_test


def test_temporal_splits_returns_none_without_dates():
    df = pd.DataFrame(
        {"text": ["a"] * 10, "label_enc": [0, 1] * 5, "date_parsed": [pd.NaT] * 10}
    )
    assert make_temporal_splits(df) is None


def test_source_splits_returns_none_without_source():
    df = pd.DataFrame({"text": ["a"] * 10, "label_enc": [0, 1] * 5})
    assert make_source_splits(df) is None


# ---------------------------------------------------------------------------
# integrity (smoke — apenas roda sem crashar)
# ---------------------------------------------------------------------------
def test_source_class_analysis_no_source_column():
    df = pd.DataFrame({"label": ["fake", "real"], "text": ["a", "b"]})
    out = source_class_analysis(df, "Test")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_length_by_class_adds_columns(synth_df):
    df = normalize_schema(synth_df)
    out = length_by_class(df, "Test")
    assert "_n_tokens" in out.columns
    assert "_n_chars" in out.columns


# ---------------------------------------------------------------------------
# ExperimentContext smoke
# ---------------------------------------------------------------------------
def test_context_defaults():
    ctx = ExperimentContext()
    assert ctx.seed == 42
    assert ctx.max_seq_len > 0
    assert ctx.device in ("cuda", "cpu")
    assert ctx.df is None
    assert isinstance(ctx.extras, dict)
    assert isinstance(ctx.results, dict)


def test_context_to_namespace_filters_none():
    ctx = ExperimentContext()
    ns = ctx.to_namespace()
    assert "DEVICE" in ns
    assert "SEED" in ns
    assert "MAX_SEQ_LEN" in ns
    assert "df" not in ns  # None não deve aparecer


def test_context_roundtrip_via_namespace():
    ctx = ExperimentContext()
    fake_df = pd.DataFrame({"a": [1, 2, 3]})
    ns = {"df": fake_df, "y_train": np.array([0, 1])}
    ctx.update_from_namespace(ns)
    assert ctx.df is fake_df
    assert ctx.y_train is not None
