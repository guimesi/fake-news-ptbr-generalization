"""Etapa 11: Paraphrasing Equalizer e equalizador linguístico.

Treina o ensemble CNN+LSTM em duas versões equalizadas do dataset (sumarização
extrativa TextRank do lado fake para igualar o comprimento médio, e remoção de
metatexto de fact-checking) e compara com o ORIGINAL para isolar viés de estilo
e de metatexto. Imprime e salva a tabela com o delta de F1.

Pré-requisitos: nenhum script anterior (treina internamente; reextrai embeddings
por variante). GPU recomendada.

Saídas (outputs/metrics/): 21_paraphrasing_equalizer.csv

Uso:
    python scripts/11_paraphrasing_equalizer.py
    python scripts/11_paraphrasing_equalizer.py --no-paraphrasing  # só sem-metatexto
    python scripts/11_paraphrasing_equalizer.py --no-metatext      # só sumarização
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import train_deep_ensemble


def _run_variant(ctx, df_variant, text_col: str, suffix: str, epochs: int = 15):
    """Treina CNN+LSTM em uma versão alternativa do dataset, devolve (acc, f1)."""
    from fakerecogna2.data.splits import make_random_splits
    from fakerecogna2.features import make_loaders
    from fakerecogna2.models import (
        TextCNN, TextLSTM, train_ensemble_on_variant,
    )

    X_tr, X_vl, X_te, y_tr, y_vl, y_te = make_random_splits(
        df_variant, text_col=text_col, seed=ctx.seed
    )
    extractor = ctx.extras["embedding_extractor"]
    emb_tr = extractor.extract_token_embs(X_tr)
    emb_vl = extractor.extract_token_embs(X_vl)
    emb_te = extractor.extract_token_embs(X_te)
    ld_tr, ld_vl, ld_te = make_loaders(emb_tr, emb_vl, emb_te, y_tr, y_vl, y_te)

    embed_dim = ctx.extras.get("embed_dim", 768)
    num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))
    return train_ensemble_on_variant(
        TextCNN, TextLSTM,
        dict(embed_dim=embed_dim, num_classes=num_classes, dropout=0.5),
        dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
             num_classes=num_classes, dropout=0.4),
        ld_tr, ld_vl, ld_te, y_te,
        device=ctx.device, epochs=epochs, suffix=suffix, seed=ctx.seed,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=15,
                   help="Épocas por variante (default 15).")
    p.add_argument("--no-paraphrasing", action="store_true", help="Pula sumarização TextRank.")
    p.add_argument("--no-metatext", action="store_true", help="Pula equalizador metatexto.")
    args = p.parse_args()

    from fakerecogna2.preprocessing import build_equalized_dataset, build_nometatext_dataset
    from fakerecogna2.utils.io_utils import RESULTS, save_table

    ctx = prepare_through_features()
    train_deep_ensemble(ctx, epochs=args.epochs)
    baseline = RESULTS["Ens2 (CNN+LSTM)"]
    rows = [{"Setup": "ORIGINAL", "Accuracy": baseline["Accuracy"], "F1": baseline["F1"]}]

    if not args.no_paraphrasing:
        df_eq = build_equalized_dataset(ctx.df)
        res = _run_variant(ctx, df_eq, "text_eq_proc", "paraphr", epochs=args.epochs)
        rows.append({"Setup": "EQUALIZADO (sumarizado)", **res})

    if not args.no_metatext:
        df_nm = build_nometatext_dataset(ctx.df)
        res = _run_variant(ctx, df_nm, "text_nometa", "nometa", epochs=args.epochs)
        rows.append({"Setup": "EQUALIZADO (sem metatexto)", **res})

    df = pd.DataFrame(rows).round(4)
    df["Δ F1"] = df["F1"].diff().fillna(0).round(4)
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    save_table(df, "21_paraphrasing_equalizer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
