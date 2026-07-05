"""Etapa 3: baselines clássicos (LogReg / LinearSVC / MLP) sobre TF-IDF e BERT[CLS].

Treina os 6 baselines (3 classificadores sklearn por 2 conjuntos de features:
TF-IDF e o vetor [CLS] do BERTimbau) e registra as métricas no cache global
`RESULTS` (outputs/.cache/results.json). Os objetos treinados ficam em
`ctx.extras` apenas em memória.

Pré-requisitos: nenhum script anterior (extrai internamente o TF-IDF e os
embeddings [CLS]). Este script não tem flags e não grava CSV próprio: a tabela
consolidada dos baselines é montada pela etapa 5 / run_all.py a partir de
`RESULTS` (e `baselines_multiseed.csv` só sai no run_all.py).

Uso:
    python scripts/03_train_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features

from fakerecogna2.models import make_baseline_configs, train_baselines


def main() -> int:
    ctx = prepare_through_features()

    configs = make_baseline_configs(
        Xtr_tf=ctx.X_train_tfidf,
        Xte_tf=ctx.X_test_tfidf,
        cls_train_np=ctx.cls_train.numpy(),
        cls_test_np=ctx.cls_test.numpy(),
        seed=ctx.seed,
    )
    trained, probs = train_baselines(configs, ctx.y_train, ctx.y_test)
    ctx.extras["baseline_configs"] = trained
    ctx.extras["baseline_probs"] = probs
    print(f"\nOK. {len(trained)} baselines treinados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
