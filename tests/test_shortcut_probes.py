"""Testes das sondas de atalho (DataFrames sintéticos, offline, sem GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fakerecogna2 import ExperimentContext
from fakerecogna2.models.shortcut_probes import (
    _official_split_indices,
    class_support_by_year,
    run_shortcut_probes,
)


def _make_ctx(n: int = 400, seed: int = 42) -> ExperimentContext:
    rng = np.random.default_rng(seed)
    # Classe 1 ("fake") com textos longos e marcador lexical; classe 0 curta.
    rows = []
    for i in range(n):
        y = i % 2
        if y == 1:
            txt = "circula redes sociais mensagem falsa " * int(rng.integers(8, 15))
        else:
            txt = "governo anuncia medida oficial " * int(rng.integers(2, 5))
        rows.append({
            "text": txt.strip(),
            "text_proc": txt.strip(),
            "label": str(y),
            "label_enc": y,
            "category": rng.choice(["politica", "saude", "outros"]),
            "date_parsed": pd.Timestamp(f"20{15 + int(rng.integers(0, 9)):02d}-06-01"),
        })
    df = pd.DataFrame(rows)

    ctx = ExperimentContext()
    ctx.df = df
    idx_tr, idx_vl, idx_te = _official_split_indices(df, seed=42)
    ctx.X_train_text = df["text_proc"].iloc[idx_tr].tolist()
    ctx.X_val_text = df["text_proc"].iloc[idx_vl].tolist()
    ctx.X_test_text = df["text_proc"].iloc[idx_te].tolist()
    ctx.y_train = df["label_enc"].to_numpy()[idx_tr]
    ctx.y_val = df["label_enc"].to_numpy()[idx_vl]
    ctx.y_test = df["label_enc"].to_numpy()[idx_te]
    ctx.extras["df_test_meta"] = df.iloc[idx_te].reset_index(drop=True)
    return ctx


def test_split_indices_are_partition():
    df = _make_ctx().df
    idx_tr, idx_vl, idx_te = _official_split_indices(df, seed=42)
    all_idx = np.concatenate([idx_tr, idx_vl, idx_te])
    assert len(all_idx) == len(df)
    assert len(np.unique(all_idx)) == len(df)
    # Proporções aproximadas 70/10/20
    assert abs(len(idx_te) / len(df) - 0.20) < 0.02


def test_probes_run_and_separate_synthetic_shortcut():
    ctx = _make_ctx()
    out = run_shortcut_probes(ctx, top_ks=(3,), save_as=None)
    assert set(out.columns) == {"probe", "n_features", "Accuracy", "F1"}
    # length + 1×top-K + metadados
    assert len(out) == 3
    # O atalho sintético de comprimento é forte: a sonda deve capturá-lo.
    length_row = out[out["probe"].str.startswith("Nº de tokens")].iloc[0]
    assert length_row["F1"] > 0.9
    # Metadados aleatórios não separam.
    meta_row = out[out["probe"].str.startswith("Metadados")].iloc[0]
    assert meta_row["F1"] < 0.7


def test_support_by_year_schema():
    ctx = _make_ctx()
    tab = class_support_by_year(ctx, save_as=None)
    assert tab is not None
    assert {"ano", "N_verdadeira", "N_fake", "N"} <= set(tab.columns)
    assert tab["N"].sum() == len(ctx.extras["df_test_meta"])
