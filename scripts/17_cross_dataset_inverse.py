"""Etapa 17 — Reexecução oficial do ramo inverso (Fake.br → FakeRecogna).

Correções em relação ao run original (ago/2026; ver CHANGELOG §14 e §16 do
artigo ENIAC):
- rótulos na convenção canônica (0=real, 1=fake) fixados antes do treino;
- pré-processamento uniformizado (Fake.br passa pelo mesmo `preprocess_base`
  do corpus principal; antes o treino usava texto cru);
- backbone BERTimbau carregado limpo (sem reaproveitar backbone ajustado de
  execuções anteriores — fragilidade apontada na auditoria da qualificação).

Regrava `17_cross_dataset_inverse_ood.csv`, `cm_ood_inverse_*` e a chave
`BERTimbau FT [Fake.br->Main]` do cache RESULTS. Backup prévio em
`outputs/metrics/_pre_rerun_inverso_2026-08/` e `outputs/backups/`.
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
    p.add_argument("--epochs-deep", type=int, default=15)
    p.add_argument("--epochs-bert", type=int, default=5)
    args = p.parse_args()

    # Dados + splits + pré-processamento (CPU). Sem embeddings do corpus
    # principal: train_on_fakebr_eval_main extrai só o que precisa.
    ctx = prepare_through_preprocessing(do_fakebr=True)
    if "df_fakebr" not in ctx.extras:
        print("Fake.br não carregado. Ajuste `data.fakebr_local_path` no config.")
        return 1

    from fakerecogna2.evaluation import train_on_fakebr_eval_main
    from fakerecogna2.features import EmbeddingExtractor

    # Backbone limpo (BERTimbau pré-treinado, sem fine-tuning prévio).
    extractor = EmbeddingExtractor(device=ctx.device, max_seq_len=ctx.max_seq_len)

    df_inv = train_on_fakebr_eval_main(
        df_fakebr=ctx.extras["df_fakebr"],
        X_main_test=ctx.X_test_text,
        y_main_test=ctx.y_test,
        extractor=extractor,
        tokenizer=extractor.tokenizer,
        bert_model=extractor.model,
        device=ctx.device,
        epochs_deep=args.epochs_deep,
        epochs_bert=args.epochs_bert,
        seed=ctx.seed,
    )
    print(df_inv.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
