"""Etapa 13 — Ablações finais (A: preprocessing, B: max_len, D: learning curve,
E: performance por faixa, Q: quartil curto).

Ablações C (extrativa) e F (equalizador linguístico) ficam nos scripts 04 e
11 respectivamente — usam treino de ensemble em variantes do dataset.

Uso::

    python scripts/13_ablations.py                   # tudo
    python scripts/13_ablations.py --only A          # só A
    python scripts/13_ablations.py --skip B,D        # pula B e D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import (
    prepare_through_data,
    prepare_through_features,
    prepare_through_preprocessing,
)
from _training import train_deep_ensemble

ABLATIONS = ("A", "B", "D", "E", "Q")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", type=str, default=None,
                   help=f"Roda só ablações listadas. Opções: {','.join(ABLATIONS)}")
    p.add_argument("--skip", type=str, default="",
                   help="Pula ablações listadas (separadas por vírgula).")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seqlen", type=int, default=300, help="max_seq_len da ablação B.")
    args = p.parse_args()

    requested = set(args.only.split(",")) if args.only else set(ABLATIONS)
    skipped = set(args.skip.split(",")) if args.skip else set()
    to_run = requested - skipped

    from fakerecogna2.evaluation import (
        learning_curve_ablation,
        performance_by_length_ensemble,
        preprocessing_ablation,
        seqlen_ablation,
        short_quartile_class_distribution,
    )

    needs_features = bool(to_run & {"B", "D", "E"})
    needs_data_only = "Q" in to_run and not needs_features and "A" not in to_run

    # Setup do contexto conforme demanda
    if needs_features:
        ctx = prepare_through_features()
    elif "A" in to_run:
        ctx = prepare_through_preprocessing()
    elif needs_data_only:
        ctx = prepare_through_data()
    else:
        return 0

    if "A" in to_run:
        preprocessing_ablation(ctx.df, n_samples=15000, seed=ctx.seed)

    if "B" in to_run:
        seqlen_ablation(ctx, seq_len=args.seqlen, epochs=args.epochs)

    if "D" in to_run:
        learning_curve_ablation(ctx, epochs=args.epochs)

    if "E" in to_run:
        train_deep_ensemble(ctx, epochs=args.epochs)
        from _training import make_predict_ens3
        _predict_ens3 = make_predict_ens3(ctx, batch_size=32)
        performance_by_length_ensemble(ctx, _predict_ens3, model_name="Ens3")

    if "Q" in to_run:
        short_quartile_class_distribution(ctx.X_test_text, ctx.y_test)

    return 0


if __name__ == "__main__":
    sys.exit(main())
