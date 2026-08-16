"""Etapa 2 — EDA textual + pré-processamento + splits + embeddings BERTimbau.

Cobre Seções 3 (integridade), 4 (log-odds), 5.1 (preprocess), 6.1 (splits) e
7 (embeddings BERTimbau) do notebook.

Uso::

    python scripts/02_preprocess_text.py                # padrão
    python scripts/02_preprocess_text.py --skip-logodds # pula análise lexical
    python scripts/02_preprocess_text.py --no-embeddings # só até preprocessing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import (
    prepare_through_features,
    prepare_through_preprocessing,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-logodds", action="store_true", help="Pula log-odds + chi².")
    p.add_argument(
        "--no-embeddings", action="store_true",
        help="Pula extração de embeddings BERTimbau (só preprocessing).",
    )
    p.add_argument("--no-extrativa", action="store_true", help="Pula variante extrativa.")
    args = p.parse_args()

    do_extrativa = not args.no_extrativa

    if args.no_embeddings:
        ctx = prepare_through_preprocessing(
            do_integrity_checks=True, do_extrativa=do_extrativa
        )
    else:
        ctx = prepare_through_features(
            do_integrity_checks=True, do_extrativa=do_extrativa
        )

    if not args.skip_logodds:
        from fakerecogna2.statistics import chi2_top, lex_analysis

        class_names = ctx.extras.get("class_names", ["fake", "real"])
        ctx.extras["lex_abst"] = lex_analysis(ctx.df, name="abstrativa")
        if "df_extr" in ctx.extras:
            ctx.extras["lex_extr"] = lex_analysis(ctx.extras["df_extr"], name="extrativa")
        ctx.extras["chi2_abst"] = chi2_top(ctx.df, class_names=class_names)

    print(f"\nOK. df={ctx.df.shape} train/val/test={len(ctx.X_train_text)}/"
          f"{len(ctx.X_val_text)}/{len(ctx.X_test_text)}")
    if ctx.token_train is not None:
        print(f"     embeddings token={tuple(ctx.token_train.shape)} "
              f"cls={tuple(ctx.cls_train.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
