"""Etapa 14: Relatório consolidado (Markdown + JSON).

Lê todos os CSVs em outputs/metrics/ e o dict RESULTS, gera relatório
consolidado. Não precisa retreinar nada, só requer que os scripts anteriores
tenham deixado artefatos em outputs/.

Uso::

    python scripts/14_generate_report.py                    # padrão
    python scripts/14_generate_report.py --with-data        # também roda data pipeline
                                                            # pra preencher metadados
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import setup_context, prepare_through_data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--with-data", action="store_true",
        help="Roda data pipeline pra preencher metadados (nº amostras, classes).",
    )
    args = p.parse_args()

    from fakerecogna2.reports import build_consolidated_report, build_results_json

    if args.with_data:
        ctx = prepare_through_data(do_extrativa=True)
        df_extr = ctx.extras.get("df_extr")
        class_names = ctx.extras.get("class_names")
        build_consolidated_report(
            df_abst=ctx.df, df_extr=df_extr,
            X_train=ctx.X_train_text, X_val=ctx.X_val_text, X_test=ctx.X_test_text,
            class_names=class_names, device=ctx.device,
        )
        build_results_json(
            df_abst=ctx.df, df_extr=df_extr,
            X_train=ctx.X_train_text, X_val=ctx.X_val_text, X_test=ctx.X_test_text,
            class_names=class_names, device=ctx.device,
        )
    else:
        # Modo light: só lê CSVs já existentes
        ctx = setup_context()
        build_consolidated_report(device=ctx.device)
        build_results_json(device=ctx.device)

    print(f"\n✅ Relatório em outputs/relatorio_final.md")
    print(f"✅ JSON em outputs/resultados.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
