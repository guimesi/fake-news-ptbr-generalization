"""Relatorio consolidado de integracao XAI <-> erros (modo standalone).

Modo standalone: usa os CSVs ja gerados pelo `run_all.py` em
`outputs/metrics/`. Nao requer recarregar modelos ou re-executar predicoes.

Cruza:
  - `error_distrib_ens3_by_categoria.csv` (e por fonte/ano/confianca)
  - `15_lime_tokens_by_tag.csv` (tokens agregados por TP/TN/FP/FN)
  - `15_lime_records.csv` (LIME por exemplo)

Produz:
  - `outputs/metrics/xai_err_consolidated_standalone.md`: relatorio narrativo
    listando categorias com taxa de erro acima da media e os tokens mais
    influentes por celula da matriz de confusao.

Para um cruzamento por exemplo (categoria x celula x token), e necessario
rodar `run_all.py` (que dispara o item 12d via `consolidate_xai_errors`),
porque as predicoes ENs3 nao sao persistidas em disco.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8")

METRICS = ROOT / "outputs" / "metrics"
REPORT = METRICS / "xai_err_consolidated_standalone.md"


def _read_csv(name: str) -> pd.DataFrame | None:
    p = METRICS / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _section_categorias_com_erro_alto(model_slug: str) -> list[str]:
    df = _read_csv(f"error_distrib_{model_slug}_by_categoria.csv")
    if df is None or df.empty:
        return [f"_(arquivo error_distrib_{model_slug}_by_categoria.csv ausente)_"]
    df = df.sort_values("Taxa erro", ascending=False)
    # Pega categorias com taxa de erro acima da mediana
    median_err = df["Taxa erro"].median()
    high = df[df["Taxa erro"] > median_err].head(15)
    lines = [
        f"Mediana da taxa de erro entre categorias: **{median_err:.4f}**.",
        f"Categorias com taxa de erro acima da mediana ({len(high)} listadas):",
        "",
        high.to_markdown(index=False),
        "",
    ]
    return lines


def _section_tokens_por_celula() -> list[str]:
    df = _read_csv("15_lime_tokens_by_tag.csv")
    if df is None or df.empty:
        return ["_(15_lime_tokens_by_tag.csv ausente; rode pipeline com XAI ativo)_"]

    lines = [
        "Pesos absolutos somados sobre os exemplos LIME (n=3 por celula).",
        "Top 10 tokens por celula da matriz de confusao:",
        "",
    ]
    for tag in ("TP", "TN", "FP", "FN"):
        sub = df[df["tag"] == tag].sort_values("abs_weight_sum", ascending=False).head(10)
        if sub.empty:
            lines.append(f"### {tag} (sem dados)\n")
            continue
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(sub[["token", "abs_weight_sum", "n_examples"]].to_markdown(index=False))
        lines.append("")
    return lines


def _section_fonte(model_slug: str) -> list[str]:
    df = _read_csv(f"error_distrib_{model_slug}_by_fonte.csv")
    if df is None or df.empty:
        return [f"_(error_distrib_{model_slug}_by_fonte.csv ausente)_"]
    df = df.sort_values("Taxa erro", ascending=False).head(15)
    return [
        "Top 15 fontes (por taxa de erro):",
        "",
        df.to_markdown(index=False),
        "",
    ]


def _section_ano(model_slug: str) -> list[str]:
    df = _read_csv(f"error_distrib_{model_slug}_by_year.csv")
    if df is None or df.empty:
        return [f"_(error_distrib_{model_slug}_by_year.csv ausente)_"]
    return [
        "Distribuicao de erros por ano:",
        "",
        df.to_markdown(index=False),
        "",
    ]


def _section_confianca(model_slug: str) -> list[str]:
    df = _read_csv(f"error_distrib_{model_slug}_by_confidence.csv")
    if df is None or df.empty:
        return [f"_(error_distrib_{model_slug}_by_confidence.csv ausente)_"]
    return [
        "Distribuicao de erros por quartil de confianca preditiva:",
        "",
        df.to_markdown(index=False),
        "",
    ]


def main() -> int:
    lines = [
        "# Relatorio consolidado - XAI <-> Erros (standalone)",
        "",
        "Gerado a partir dos CSVs em `outputs/metrics/`, sem re-executar o",
        "pipeline. Para um cruzamento exemplo-por-exemplo (categoria x",
        "celula x token), execute `run_all.py` que dispara a etapa 12d com",
        "`consolidate_xai_errors`.",
        "",
        "---",
        "",
        "## 1. Categorias com taxa de erro acima da mediana - Ens3 (CNN+LSTM+ConvLSTM)",
        "",
    ]
    lines += _section_categorias_com_erro_alto("ens3")
    lines += [
        "## 2. Categorias com taxa de erro acima da mediana - BERTimbau FT",
        "",
    ]
    lines += _section_categorias_com_erro_alto("bert_ft")
    lines += [
        "## 3. Tokens LIME mais influentes por celula da matriz de confusao",
        "",
    ]
    lines += _section_tokens_por_celula()
    lines += [
        "## 4. Distribuicao de erros por fonte - Ens3",
        "",
    ]
    lines += _section_fonte("ens3")
    lines += [
        "## 5. Distribuicao de erros por ano - Ens3",
        "",
    ]
    lines += _section_ano("ens3")
    lines += [
        "## 6. Distribuicao de erros por confianca preditiva - Ens3",
        "",
    ]
    lines += _section_confianca("ens3")
    lines += [
        "---",
        "",
        "## Interpretacao orientativa",
        "",
        "Use este relatorio para responder perguntas como:",
        "",
        "1. **Quais categorias concentram erros?** Cruze a distribuicao por",
        "   categoria com as de ano e fonte para verificar se uma categoria",
        "   especifica esta associada a uma fonte/ano com cobertura desigual.",
        "",
        "2. **Os tokens LIME refletem viseis de dominio?** Se TP/TN (acertos)",
        "   priorizam tokens semanticamente plausiveis (ex.: \"verificou\",",
        "   \"checagem\") e FP/FN (erros) priorizam tokens generales ou",
        "   metatextuais, isso aponta para confianca em marcas de origem",
        "   em vez de conteudo discriminativo.",
        "",
        "3. **Erros por confianca:** se quartis baixos de confianca",
        "   concentram mais erros que altos, isso e calibracao saudavel.",
        "   Se erros aparecem em alta confianca, ha *overconfident wrong*,",
        "   o que requer revisao de calibracao (ver `13_calibration.csv`).",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] Relatorio salvo: {REPORT}")
    print(f"     {len(lines)} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
