"""Etapa 3 — Baselines clássicos (LogReg / LinearSVC / MLP) em TF-IDF e BERT[CLS].

Cobre a Seção 8 do notebook (cell 29).
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
