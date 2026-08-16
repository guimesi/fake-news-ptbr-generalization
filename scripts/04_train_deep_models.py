"""Etapa 4 — Modelos deep (CNN/LSTM/ConvLSTM multi-seed) + ensembles + BERTimbau FT.

Cobre Seções 9 (arquiteturas + treino), 10 (ensembles), 11 (BERT FT) e 12
(ensemble extrativa) do notebook.

Uso::

    python scripts/04_train_deep_models.py                    # padrão
    python scripts/04_train_deep_models.py --no-bert          # pula BERT FT
    python scripts/04_train_deep_models.py --epochs 5         # debug rápido
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import train_bert_classifier, train_deep_ensemble


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-bert", action="store_true", help="Pula BERTimbau FT.")
    p.add_argument("--epochs", type=int, default=30, help="Épocas dos modelos deep.")
    p.add_argument("--bert-epochs", type=int, default=10)
    args = p.parse_args()

    ctx = prepare_through_features()
    train_deep_ensemble(ctx, epochs=args.epochs)

    if not args.no_bert:
        train_bert_classifier(ctx, epochs=args.bert_epochs)

    print(f"\nOK. {len(ctx.predictions)} modelos treinados/predições registradas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
