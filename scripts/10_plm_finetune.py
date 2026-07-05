"""Etapa 10: fine-tuning de PLMs maiores (BERTimbau-large, XLM-R, mDeBERTa-v3).

Faz o fine-tuning e a avaliação dos três PLMs candidatos definidos em
`models.plm_candidates` (configs/config.yaml). Rodado isolado, este script usa
uma única seed; o run_all.py roda multi-seed via `--trans-seeds`.

Pré-requisitos: nenhum script anterior (só precisa de textos e labels, não dos
embeddings BERTimbau base). Baixa vários GB de modelos na primeira execução
(BERTimbau-large ~1.3 GB, XLM-R ~1.1 GB, mDeBERTa ~0.7 GB). GPU fortemente
recomendada.

Saídas (outputs/metrics/): 20_larger_models.csv (e plms_multiseed.csv no caminho
multi-seed), mais as matrizes de confusão por PLM.

Uso:
    python scripts/10_plm_finetune.py
    python scripts/10_plm_finetune.py --epochs 4 --es-patience 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_preprocessing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=4,
                   help="Épocas de fine-tuning por PLM (default 4).")
    p.add_argument("--es-patience", type=int, default=2,
                   help="Paciência do early stopping (default 2).")
    args = p.parse_args()

    # PLMs precisam só de textos e labels, não dos embeddings BERTimbau-base
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
        device=ctx.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
