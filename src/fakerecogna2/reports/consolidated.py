"""Tabela consolidada + relatório Markdown/JSON.

Cobre as células 53, 55, 85, 86, 87 (Seções 16 e 23) e 116 (Seção K).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from tabulate import tabulate

from ..config import ARTIFACTS_DIR, BATCH_SIZE, MAX_LEN, SEED, SEEDS_MULTI
from ..utils.io_utils import RESULTS, RESULTS_MULTISEED, save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- Helpers de I/O -----------------------------------------------------------
def _read_csv_if_exists(name: str) -> pd.DataFrame | None:
    p = ARTIFACTS_DIR / "metrics" / f"{name}.csv"
    if not p.exists():
        p = ARTIFACTS_DIR / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _df_to_md(df: pd.DataFrame | None, max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_(tabela não gerada nesta execução)_"
    if max_rows is not None:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


# -- Tabela final consolidada -------------------------------------------------
def build_final_table(
    results: dict | None = None,
    bootstrap_ci_df: pd.DataFrame | None = None,
    save_as: str | None = "16_final_results",
) -> pd.DataFrame:
    """Tabela de resultados ordenada por F1 desc, com IC bootstrap quando disponível."""
    results = results if results is not None else RESULTS
    df_final = pd.DataFrame(results).T
    df_final.index.name = "Modelo"
    df_final = df_final.sort_values("F1", ascending=False)

    if bootstrap_ci_df is None:
        bootstrap_ci_df = _read_csv_if_exists("13_bootstrap_ci")

    ci_map: dict[str, tuple[float, float]] = {}
    if bootstrap_ci_df is not None and not bootstrap_ci_df.empty:
        ci_map = dict(
            zip(
                bootstrap_ci_df["Model"],
                zip(bootstrap_ci_df["F1_95ci_lo"], bootstrap_ci_df["F1_95ci_hi"]),
            )
        )

    def _fmt_f1(row) -> str:
        base = f"{row['F1']*100:.2f}%"
        if row.name in ci_map:
            lo, hi = ci_map[row.name]
            return f"{base}  [{lo*100:.2f}–{hi*100:.2f}]"
        return base

    df_disp = df_final.copy()
    df_disp["F1 (95% CI)"] = df_disp.apply(_fmt_f1, axis=1)
    for c in ["Accuracy", "Precision", "Recall", "F1"]:
        if c in df_disp.columns:
            df_disp[c] = (df_disp[c] * 100).round(2).astype(str) + "%"
    if "Inference (ms)" in df_disp.columns:
        df_disp["Inference (ms)"] = df_disp["Inference (ms)"].round(2)

    print("\n" + "=" * 90)
    print("RESULTADOS FINAIS CONSOLIDADOS")
    print("=" * 90)
    print(tabulate(df_disp, headers="keys", tablefmt="github"))

    if save_as is not None:
        save_table(df_final.reset_index(), save_as)
    log.info(
        f'Melhor F1: {df_final["F1"].idxmax()} '
        f'({df_final["F1"].max()*100:.2f}%)'
    )
    return df_final


def print_classification_report(
    y_test, predictions, class_names: list[str], model_name: str = "Ensemble"
) -> None:
    """Imprime sklearn.metrics.classification_report."""
    print(f"\n=== Classification Report: {model_name} ===")
    print(
        classification_report(
            y_test, predictions, target_names=[str(c) for c in class_names], digits=4
        )
    )


def build_manifest(
    n_samples: int,
    class_names: list[str],
    device: str,
    out_path: str | Path | None = None,
) -> str:
    """Gera manifesto com listagem de todos os arquivos em ARTIFACTS_DIR."""
    lines = [
        f"Manifesto — {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Dataset: FakeRecogna 2.0  |  Dispositivo: {device}",
        f"Classes: {class_names}",
        f"Número de amostras finais (abstrativa): {n_samples}",
        "",
        "--- Artefatos gerados ---",
    ]
    for root, _, files in os.walk(ARTIFACTS_DIR):
        for f in sorted(files):
            fp = Path(root) / f
            lines.append(
                f"  {fp.relative_to(ARTIFACTS_DIR)} ({fp.stat().st_size/1024:.1f} KB)"
            )
    out_path = Path(out_path) if out_path else (ARTIFACTS_DIR / "manifest.txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:15] + ["..."]))
    return "\n".join(lines)


# -- Relatório Markdown -------------------------------------------------------
def build_consolidated_report(
    out_path: str | Path | None = None,
    df_abst=None,
    df_extr=None,
    X_train: list | None = None,
    X_val: list | None = None,
    X_test: list | None = None,
    class_names: list[str] | None = None,
    device: str = "cpu",
    dedupe_stats: dict | None = None,
    results: dict | None = None,
) -> str:
    """Gera relatório Markdown lendo CSVs em outputs/metrics/ + dict RESULTS."""
    if out_path is None:
        out_path = ARTIFACTS_DIR / "relatorio_final.md"
    out_path = Path(out_path)
    results = results if results is not None else RESULTS

    lines: list[str] = []
    lines.append("# Relatório Consolidado — Detecção de Fake News em PT-BR")
    lines.append("")
    lines.append(f"**Data de execução:** {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"**Dispositivo:** {device}")
    lines.append(f"**Seed principal:** {SEED}")
    lines.append(f"**Seeds multi-run:** {SEEDS_MULTI}")
    lines.append("")

    # 1. Caracterização do dataset
    if df_abst is not None:
        lines.append("## 1. Caracterização do dataset após limpeza")
        lines.append("")
        lines.append(f"- FakeRecogna 2.0 Abstrativa: **{len(df_abst)}** amostras após dedupe")
        if df_extr is not None:
            lines.append(f"- FakeRecogna 2.0 Extrativa:  **{len(df_extr)}** amostras após dedupe")
        if dedupe_stats:
            lines.append(
                f"- Duplicatas exatas removidas: abst={dedupe_stats.get('n_ex_a','?')}, "
                f"extr={dedupe_stats.get('n_ex_e','?')}"
            )
            lines.append(
                f"- Near-duplicates removidos: abst={dedupe_stats.get('n_nr_a','?')}, "
                f"extr={dedupe_stats.get('n_nr_e','?')}"
            )
        if class_names:
            lines.append(f"- Classes: {class_names}")
        if X_train is not None and X_val is not None and X_test is not None:
            lines.append(
                f"- Split principal: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )
        lines.append("")

    # 2. EDA lexical (log-odds)
    if df_abst is not None and "label" in df_abst.columns:
        lines.append("## 2. Top tokens discriminativos por classe (log-odds Dirichlet)")
        cls_pair = sorted(df_abst["label"].unique())
        if len(cls_pair) == 2:
            cls_a, cls_b = cls_pair
            for n in [1, 2, 3]:
                lines.append("")
                lines.append(f"### n={n}-gramas — top 10 por classe (abstrativa)")
                top_a = _read_csv_if_exists(f"04_logodds_abstrativa_n{n}_top_{cls_a}")
                top_b = _read_csv_if_exists(f"04_logodds_abstrativa_n{n}_top_{cls_b}")
                for cls_name, top_df in [(cls_a, top_a), (cls_b, top_b)]:
                    if top_df is not None:
                        lines.append("")
                        lines.append(f'**Top associados a "{cls_name}"**:')
                        lines.append("")
                        lines.append(
                            _df_to_md(top_df[["term", "count_a", "count_b", "z"]].head(10))
                        )

    # 3. Tabela consolidada IID
    if results:
        lines.append("")
        lines.append("## 3. Resultados IID — random split da Abstrativa")
        lines.append("")
        df_final = pd.DataFrame(results).T.round(4)
        df_final.index.name = "Modelo"
        df_final = df_final.sort_values("F1", ascending=False).reset_index()
        lines.append(_df_to_md(df_final))
        best_model = df_final.iloc[0]["Modelo"]
        best_f1 = df_final.iloc[0]["F1"]
        lines.append("")
        lines.append(f"**Melhor modelo em IID:** `{best_model}` com F1={best_f1:.4f}")

        df_boot = _read_csv_if_exists("13_bootstrap_ci")
        if df_boot is not None:
            lines.append("")
            lines.append("### Intervalos de confiança bootstrap (percentil 95%)")
            lines.append("")
            lines.append(_df_to_md(df_boot))

    # 4. Avaliação estatística
    lines.append("")
    lines.append("## 4. Avaliação estatística")
    for table, header in [
        ("13_mcnemar_pairwise_holm", "### McNemar pairwise (Holm)"),
        ("13_calibration", "### Calibração (ECE e Brier)"),
        ("13_cross_validation", "### 5-fold cross-validation"),
    ]:
        t = _read_csv_if_exists(table)
        if t is not None:
            lines.append("")
            lines.append(header)
            lines.append("")
            if "sig@0.05" in t.columns:
                t = t[t["sig@0.05"] == True].head(15)  # noqa: E712
            lines.append(_df_to_md(t))

    # 5. Stress tests
    lines.append("")
    lines.append("## 5. Stress tests de generalização")
    for table, header in [
        ("14_source_split", "### Split por fonte (out-of-distribution)"),
        ("14_ner_masking", "### Mascaramento de entidades nomeadas"),
        ("14_temporal_split", "### Split temporal"),
    ]:
        t = _read_csv_if_exists(table)
        if t is not None:
            lines.append("")
            lines.append(header)
            lines.append("")
            lines.append(_df_to_md(t))

    # 6-10
    for table, section in [
        ("17_cross_dataset_ood", "## 6. Avaliação cross-dataset (Fake.br-Corpus)"),
        ("19_adversarial_robustness", "## 7. Robustez adversarial"),
        ("20_larger_models", "## 8. Modelos maiores (PLMs)"),
        ("21_paraphrasing_equalizer", "## 9. Paraphrasing Equalizer"),
        ("22_deployment_metrics", "## 10. Métricas de deployment"),
    ]:
        t = _read_csv_if_exists(table)
        if t is not None:
            lines.append("")
            lines.append(section)
            lines.append("")
            lines.append(_df_to_md(t))

    # 11. Ablações
    lines.append("")
    lines.append("## 11. Ablações")
    for table_name, title in [
        ("preprocessing_ablation", "Pré-processamento (base / stem / lemma)"),
        ("seqlen_ablation", "Comprimento de sequência"),
        ("abstrativa_vs_extrativa", "Abstrativa vs Extrativa"),
        ("learning_curve", "Curva de aprendizado"),
        ("random_vs_temporal", "Random vs Temporal split"),
        ("performance_by_length", "Desempenho por faixa de comprimento"),
    ]:
        t = _read_csv_if_exists(table_name)
        if t is not None:
            lines.append("")
            lines.append(f"### {title}")
            lines.append("")
            lines.append(_df_to_md(t))

    lines.append("")
    lines.append("---")
    lines.append(f"_Relatório gerado automaticamente em {datetime.now():%Y-%m-%d %H:%M:%S}._")
    lines.append(f"_Todas as tabelas originais em `{ARTIFACTS_DIR}/metrics/`._")

    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    log.info(f"Relatório consolidado salvo: {out_path} ({len(report)} chars)")
    return report


def _collect_lexical_top_features(df_abst) -> dict | None:
    """Lê as CSVs de log-odds e devolve top-20 por (n, classe). None se ausente."""
    if df_abst is None or "label" not in df_abst.columns:
        return None
    classes = sorted(df_abst["label"].unique())
    if len(classes) != 2:
        return None
    out: dict[str, list[dict] | None] = {}
    for n in (1, 2, 3):
        for cls in classes:
            df = _read_csv_if_exists(f"04_logodds_abstrativa_n{n}_top_{cls}")
            key = f"n{n}_top_{cls}"
            out[key] = df.head(20).to_dict(orient="records") if df is not None else None
    return out


# -- Relatório JSON -----------------------------------------------------------
def build_results_json(
    out_path: str | Path | None = None,
    df_abst=None,
    df_extr=None,
    X_train: list | None = None,
    X_val: list | None = None,
    X_test: list | None = None,
    class_names: list[str] | None = None,
    device: str = "cpu",
    results: dict | None = None,
    results_multiseed: dict | None = None,
) -> dict:
    """Exporta todos os números em JSON estruturado."""
    if out_path is None:
        out_path = ARTIFACTS_DIR / "resultados.json"
    out_path = Path(out_path)
    results = results if results is not None else RESULTS
    results_multiseed = (
        results_multiseed if results_multiseed is not None else RESULTS_MULTISEED
    )

    # Se RESULTS_MULTISEED estiver vazio (ex.: rebuild a partir de CSVs),
    # tenta carregar de outputs/metrics/iid_multiseed.csv.
    if not results_multiseed:
        ms_csv = _read_csv_if_exists("iid_multiseed")
        if ms_csv is not None:
            results_multiseed = {}
            for _, row in ms_csv.iterrows():
                model = row["Model"]
                results_multiseed[model] = {}
                for col in ms_csv.columns:
                    if col.endswith("_mean"):
                        metric = col[:-5]
                        std_col = f"{metric}_std"
                        if std_col in ms_csv.columns:
                            results_multiseed[model][metric] = (
                                float(row[col]), float(row[std_col]),
                            )

    def _csv_to_records(name: str):
        df = _read_csv_if_exists(name)
        return df.to_dict(orient="records") if df is not None else None

    output = {
        "meta": {
            "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            "device": str(device),
            "seed": SEED,
            "seeds_multi": SEEDS_MULTI,
            "max_seq_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "n_abstrativa": len(df_abst) if df_abst is not None else None,
            "n_extrativa": len(df_extr) if df_extr is not None else None,
            "classes": class_names,
            "split_sizes": (
                {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
                if X_train is not None and X_val is not None and X_test is not None
                else None
            ),
        },
        "iid_results": results,
        "iid_multiseed": {
            k: {m: list(v) for m, v in inner.items()}
            for k, inner in results_multiseed.items()
        },
        "bootstrap_ci": _csv_to_records("13_bootstrap_ci"),
        "mcnemar_pairwise": _csv_to_records("13_mcnemar_pairwise_holm"),
        "calibration": _csv_to_records("13_calibration"),
        "cross_validation": _csv_to_records("13_cross_validation"),
        "stress_tests": {
            "source_split": _csv_to_records("14_source_split"),
            "ner_masking": _csv_to_records("14_ner_masking"),
            "temporal_split": _csv_to_records("14_temporal_split"),
        },
        "cross_dataset": {
            "ood_table": _csv_to_records("17_cross_dataset_ood"),
            "delta_table": _csv_to_records("17_cross_dataset_delta"),
        },
        "adversarial": _csv_to_records("19_adversarial_robustness"),
        "larger_models": _csv_to_records("20_larger_models"),
        "paraphrasing_equalizer": _csv_to_records("21_paraphrasing_equalizer"),
        "deployment": _csv_to_records("22_deployment_metrics"),
        "ablations": {
            "preprocessing": _csv_to_records("preprocessing_ablation"),
            "seqlen": _csv_to_records("seqlen_ablation"),
            "abst_vs_ext": _csv_to_records("abstrativa_vs_extrativa"),
            "learning_curve": _csv_to_records("learning_curve"),
            "random_vs_temporal": _csv_to_records("random_vs_temporal"),
            "by_length": _csv_to_records("performance_by_length"),
        },
        "lexical_top_features": _collect_lexical_top_features(df_abst),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"JSON de resultados salvo: {out_path}")
    return output


__all__ = [
    "build_final_table",
    "print_classification_report",
    "build_manifest",
    "build_consolidated_report",
    "build_results_json",
]
