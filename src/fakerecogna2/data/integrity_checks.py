"""Checagens de integridade do corpus: fonte por classe, comprimento, temporal, NER.

Inclui:
- Análise de FONTE por classe (detecção de atalho estilométrico)
- Distribuição de COMPRIMENTO por classe (Mann-Whitney U)
- Distribuição TEMPORAL por classe
- Top entidades nomeadas (NER) por classe

Funções com side-effects controlados: imprimem sumários no log e salvam
plots via `save_plot`. Retornam DataFrames/dicts para inspeção posterior.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from ..config import SEED
from ..utils.io_utils import save_plot
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def source_class_analysis(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Top-20 fontes × classes; identifica fontes exclusivas (risco de atalho).

    Args:
        df: DataFrame com colunas 'source' e 'label'.
        name: rótulo da variante pra logging (ex.: "Abstrativa").

    Returns:
        DataFrame com fontes exclusivas a 1 classe (≥5 amostras). Vazio se nenhuma.
    """
    if "source" not in df.columns:
        log.info(f"{name}: sem coluna source — pulando")
        return pd.DataFrame()

    pv = pd.crosstab(df["source"], df["label"])
    pv["total"] = pv.sum(1)
    pv = pv.sort_values("total", ascending=False).head(20)
    print(f"\n=== {name}: Top-20 fontes × classes ===")
    print(pv.to_string())

    excl = []
    for src, row in pd.crosstab(df["source"], df["label"]).iterrows():
        nz = (row > 0).sum()
        if nz == 1 and row.sum() >= 5:
            cls = row.idxmax()
            excl.append({"source": src, "class": cls, "n": int(row.max())})

    dfe = pd.DataFrame(excl).sort_values("n", ascending=False) if excl else pd.DataFrame()
    if not dfe.empty:
        log.warning(
            f"{name}: {len(dfe)} fontes exclusivas de 1 classe "
            f'({dfe["n"].sum()} amostras)'
        )
        print("\nFontes com ≥5 amostras E exclusivas a 1 classe (risco de atalho):")
        print(dfe.head(15).to_string(index=False))
    return dfe


def length_by_class(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Distribuição de comprimento (tokens/chars) por classe.

    Adiciona colunas `_n_tokens` e `_n_chars`. Imprime sumário e roda
    teste Mann-Whitney U (significativo = comprimento correlacionado a classe
    → risco de atalho).
    """
    df = df.copy()
    df["_n_tokens"] = df["text"].astype(str).str.split().str.len()
    df["_n_chars"] = df["text"].astype(str).str.len()
    by_cls = df.groupby("label").agg(
        n=("_n_tokens", "count"),
        tokens_mean=("_n_tokens", "mean"),
        tokens_median=("_n_tokens", "median"),
        tokens_p90=("_n_tokens", lambda s: s.quantile(0.9)),
        chars_mean=("_n_chars", "mean"),
    )
    print(f"\n=== {name}: Comprimento por classe ===")
    print(by_cls.round(1).to_string())

    classes = df["label"].unique()
    if len(classes) == 2:
        a = df[df["label"] == classes[0]]["_n_tokens"]
        b = df[df["label"] == classes[1]]["_n_tokens"]
        _u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        log.info(
            f"{name}: Mann-Whitney U no comprimento — p={p:.3e} "
            "(p<.05 = comprimento correlacionado à classe → risco de atalho)"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls in df["label"].unique():
        sub = df[df["label"] == cls]["_n_tokens"]
        ax.hist(sub.clip(0, 1500), bins=50, alpha=0.55, label=str(cls))
    ax.set_title(f"{name} — tokens por classe")
    ax.set_xlabel("# tokens")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f"03_length_per_class_{name.lower()}")
    return df


def plot_length_per_class_combined(
    df_abst: pd.DataFrame,
    df_extr: pd.DataFrame,
    save_as: str = "03_length_per_class_combined",
) -> None:
    """Plota distribuição de tokens por classe em Abstrativa e Extrativa lado a lado.

    Corresponde à Figura 2 do Cap. 5 da dissertação. Cada subplot mostra,
    para uma variante, dois histogramas (uma cor por classe) compartilhando
    o mesmo eixo de tokens (clipado em 1500 para excluir outliers).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, df, name in [
        (axes[0], df_abst, "Abstrativa"),
        (axes[1], df_extr, "Extrativa"),
    ]:
        if "_n_tokens" not in df.columns:
            df = df.copy()
            df["_n_tokens"] = df["text"].astype(str).str.split().str.len()
        for cls in sorted(df["label"].unique()):
            sub = df[df["label"] == cls]["_n_tokens"]
            ax.hist(sub.clip(0, 1500), bins=50, alpha=0.55, label=str(cls))
        ax.set_title(f"{name} — tokens por classe")
        ax.set_xlabel("# tokens")
        ax.legend()
    axes[0].set_ylabel("# documentos")
    plt.tight_layout()
    save_plot(fig, save_as)


def temporal_analysis(df: pd.DataFrame, name: str = "Abstrativa") -> None:
    """Plota distribuição temporal por classe. Salva figura via save_plot."""
    if df["date_parsed"].notna().sum() <= len(df) * 0.5:
        log.warning("Distribuição temporal inconclusiva (<50% de datas válidas).")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in df["label"].unique():
        sub = df[df["label"] == cls].dropna(subset=["date_parsed"])
        sub.groupby(sub["date_parsed"].dt.to_period("M")).size().plot(
            ax=ax, label=str(cls), marker="."
        )
    ax.set_title(f"Distribuição temporal por classe ({name})")
    ax.set_ylabel("# amostras/mês")
    ax.legend()
    save_plot(fig, "03_temporal_per_class")

    for cls in df["label"].unique():
        sub = df[df["label"] == cls].dropna(subset=["date_parsed"])
        log.info(
            f"{cls}: {sub['date_parsed'].min().date()} → "
            f"{sub['date_parsed'].max().date()} (n={len(sub)})"
        )


def ner_top_by_class(
    df: pd.DataFrame,
    nlp,
    name: str,
    k: int = 15,
    n_sample: int = 2000,
    labels: tuple[str, ...] = ("PER", "ORG", "LOC", "MISC"),
) -> dict[str, Counter]:
    """Top entidades nomeadas por classe (amostragem).

    Args:
        df: DataFrame com colunas 'text' e 'label'.
        nlp: pipeline spaCy carregado (ex.: `spacy.load('pt_core_news_sm')`).
        name: rótulo da variante pra logging.
        k: top-K entidades a imprimir por classe.
        n_sample: tamanho da amostra (reduz custo, NER em 25k textos é lento).
        labels: tipos de entidade a contar.
    """
    samp = df.sample(min(n_sample, len(df)), random_state=SEED)
    by_cls: dict = defaultdict(Counter)
    for txt, cls in tqdm(
        zip(samp["text"], samp["label"]),
        total=len(samp),
        desc=f"NER {name}",
        leave=False,
    ):
        doc = nlp(str(txt)[:3000])
        for ent in doc.ents:
            if ent.label_ in labels:
                by_cls[cls][(ent.text.strip(), ent.label_)] += 1

    print(f"\n=== {name}: Top-{k} entidades por classe ===")
    for cls, cnt in by_cls.items():
        print(f"\n-- {cls} --")
        for (ent, lab), n in cnt.most_common(k):
            print(f"  [{lab}] {ent}: {n}")
    return dict(by_cls)


__all__ = [
    "source_class_analysis",
    "length_by_class",
    "plot_length_per_class_combined",
    "temporal_analysis",
    "ner_top_by_class",
]
