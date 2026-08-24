"""Etapa 10 — PLMs maiores (BERTimbau-large, XLM-R, mDeBERTa) — Seção 20."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_preprocessing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--es-patience", type=int, default=2)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 7],
        help="Sementes do multi-seed (default [42, 7], igual ao run_all.py; "
        "a primeira é a principal, usada em CM/bootstrap).",
    )
    args = p.parse_args()

    # PLMs precisam só de textos e labels — não dos embeddings BERTimbau-base
    ctx = prepare_through_preprocessing()

    from fakerecogna2.models import evaluate_plm_candidates

    evaluate_plm_candidates(
        X_train=ctx.X_train_text, y_train=ctx.y_train,
        X_val=ctx.X_val_text, y_val=ctx.y_val,
        X_test=ctx.X_test_text, y_test=ctx.y_test,
        num_classes=len(ctx.extras.get("class_names", ["fake", "real"])),
        max_seq_len=ctx.max_seq_len,
        epochs=args.epochs,
        es_patience=args.es_patience,
        seeds=args.seeds,
        device=ctx.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
