"""Etapa 1 — Carregamento + integridade + preprocessing + splits.

Carrega FakeRecogna 2.0, faz checagens de integridade do corpus, aplica
preprocessing base e gera splits estratificados sobre `text_proc`.

Uso::

    python scripts/01_prepare_data.py                     # padrão
    python scripts/01_prepare_data.py --ner               # também roda NER (lento)
    python scripts/01_prepare_data.py --fakebr            # também carrega Fake.br
    python scripts/01_prepare_data.py --skip-integrity    # pula checagens de integridade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_preprocessing

from fakerecogna2.utils import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner", action="store_true", help="Roda NER (lento, ~2 min CPU).")
    parser.add_argument(
        "--fakebr", action="store_true",
        help="Também carrega Fake.br-Corpus (path configurado em configs/config.yaml).",
    )
    parser.add_argument(
        "--no-extrativa", action="store_true",
        help="Pula a variante extrativa (usada como controle).",
    )
    parser.add_argument(
        "--skip-integrity", action="store_true",
        help="Pula análises da Seção 3 (fonte/comprimento/temporal).",
    )
    args = parser.parse_args()

    log = get_logger()
    ctx = prepare_through_preprocessing(
        do_integrity_checks=not args.skip_integrity,
        do_ner=args.ner,
        do_extrativa=not args.no_extrativa,
        do_fakebr=args.fakebr,
    )

    log.info(
        "Dados prontos: "
        f"df={ctx.df.shape} train={len(ctx.X_train_text)} "
        f"val={len(ctx.X_val_text)} test={len(ctx.X_test_text)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
