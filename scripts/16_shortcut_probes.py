"""Etapa 16 — Sondas de atalho (classificadores triviais).

Quantifica quanto do desempenho IID é recuperável por sinais superficiais:
apenas comprimento, apenas top-K termos discriminativos (log-odds no treino),
apenas metadados (categoria+ano). Motivada pela revisão do ENIAC 2026.

Salva `outputs/metrics/23_shortcut_probes.csv` e
`outputs/metrics/23_test_support_by_year.csv` (suporte classe×ano no teste,
base da leitura da tabela temporal do artigo). CPU-only, sem embeddings.
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
    p.add_argument(
        "--top-ks", type=int, nargs="+", default=[6, 20, 50],
        help="valores de K para a sonda de top-K termos por classe",
    )
    args = p.parse_args()

    ctx = prepare_through_preprocessing()

    from fakerecogna2.models.shortcut_probes import (
        class_support_by_year,
        run_shortcut_probes,
    )

    df = run_shortcut_probes(ctx, top_ks=tuple(args.top_ks))
    print(df.to_string(index=False))
    class_support_by_year(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
