"""Etapa 9: robustez adversarial.

Aplica quatro perturbações no test set (typos com mapa de teclado, deleção de
palavras, troca de palavras adjacentes e back-translation PT para EN para PT via
MarianMT) e mede a queda de F1 do Ens3 e do BERTimbau FT.

Pré-requisitos: nenhum script anterior (treina internamente). A back-translation
baixa os modelos MarianMT na primeira execução.

Saídas (outputs/metrics/): 19_adversarial_robustness.csv

Uso:
    python scripts/09_adversarial_robustness.py
    python scripts/09_adversarial_robustness.py --no-bt          # pula back-translation
    python scripts/09_adversarial_robustness.py --n-sample 200   # amostra menor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import train_bert_classifier, train_deep_ensemble


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-bt", action="store_true", help="Pula back-translation (lento).")
    p.add_argument("--n-sample", type=int, default=500,
                   help="Tamanho da amostra do test set por perturbação (default 500).")
    args = p.parse_args()

    ctx = prepare_through_features()
    train_deep_ensemble(ctx)
    train_bert_classifier(ctx)

    from _training import make_predict_bert_ft, make_predict_ens3
    predict_ens3 = make_predict_ens3(ctx, batch_size=32)
    predict_bert_ft = make_predict_bert_ft(ctx, batch_size=32)

    from fakerecogna2.adversarial import run_adversarial_eval

    run_adversarial_eval(
        ctx.X_test_text, ctx.y_test,
        models={"Ens3": predict_ens3, "BERT FT": predict_bert_ft},
        n_sample=args.n_sample,
        do_back_translation=not args.no_bt,
        device=ctx.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
