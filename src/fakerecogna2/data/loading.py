"""Carregamento e normalização do FakeRecogna 2.0.

Etapas:
- Carregamento do HuggingFace Hub
- Mapeamento determinístico de colunas
- Deduplicação exata (SHA-1) + near-duplicate (MinHashLSH)

Funções aqui são puras quando possível: recebem DataFrames, retornam novos.
Side-effects (logging) são mantidos em nível INFO para visibilidade durante
execução do pipeline.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from ..config import (
    DATASET_HF_ID_TEMPLATE,
    DEDUP_NUM_PERM,
    DEDUP_SHINGLE_SIZE,
    DEDUP_THRESHOLD,
)
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def load_fakerecogna(variant: str) -> pd.DataFrame:
    """Carrega uma variante do FakeRecogna 2.0 (HuggingFace).

    Args:
        variant: 'abstrativa' (real reescrita por PTT5) ou 'extrativa'
            (real com sentenças extraídas). Em ambos os casos as fakes
            vêm originais das agências de fact-checking.

    Returns:
        DataFrame com todos os splits concatenados e coluna `_variant`.
    """
    ds = load_dataset(
        DATASET_HF_ID_TEMPLATE.format(variant=variant), trust_remote_code=True
    )
    parts = [ds[s].to_pandas() for s in ds.keys()]
    df = pd.concat(parts, ignore_index=True)
    df["_variant"] = variant
    log.info(f"{variant}: {df.shape} colunas={list(df.columns)}")
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para schema canônico (lowercase).

    Mapeia variações em português/inglês/maiúsculas para:
    `text, title, subtitle, category, author, date, url, label`.
    Deriva também `source` extraindo o domínio da URL.
    """
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("news", "text", "texto", "noticia", "body", "content"):
            rename[c] = "text"
        elif cl in ("title", "titulo", "título"):
            rename[c] = "title"
        elif cl in ("subtitle", "subtitulo", "subtítulo"):
            rename[c] = "subtitle"
        elif cl in ("category", "categoria"):
            rename[c] = "category"
        elif cl in ("author", "autor"):
            rename[c] = "author"
        elif cl in ("date", "data"):
            rename[c] = "date"
        elif cl in ("url", "link", "source_url"):
            rename[c] = "url"
        elif cl in ("label", "classe", "rotulo", "fake"):
            rename[c] = "label"
    df = df.rename(columns=rename)
    if "url" in df.columns:
        df["source"] = (
            df["url"]
            .astype(str)
            .str.extract(r"https?://(?:www\.)?([^/]+)/?", expand=False)
        )
    return df


def encode_labels(
    df: pd.DataFrame, encoder: LabelEncoder | None = None
) -> tuple[pd.DataFrame, LabelEncoder, list[str]]:
    """Encoda label fake/real → 0/1.

    Args:
        df: DataFrame com coluna `label`.
        encoder: se fornecido, reusa (mantém mapping consistente entre
            variantes). Caso contrário, fita um novo.

    Returns:
        (df_com_label_enc, encoder, class_names)
    """
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df["label_enc"] = encoder.fit_transform(df["label"].astype(str))
    else:
        df["label_enc"] = encoder.transform(df["label"].astype(str))
    return df, encoder, list(encoder.classes_)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Faz parse robusto da coluna `date` (formato brasileiro dd/mm/yyyy)."""
    df = df.copy()
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        v = df["date_parsed"].notna().sum()
        log.info(f"Datas parseadas: {v}/{len(df)} ({100 * v / len(df):.1f}%)")
    else:
        df["date_parsed"] = pd.NaT
        log.warning("Coluna 'date' ausente — split temporal será desabilitado.")
    return df


def dedupe(
    df: pd.DataFrame,
    text_col: str = "text",
    thr: float = DEDUP_THRESHOLD,
    num_perm: int = DEDUP_NUM_PERM,
    shingle: int = DEDUP_SHINGLE_SIZE,
) -> tuple[pd.DataFrame, int, int]:
    """Remove duplicatas exatas (SHA-1) e near-duplicates (MinHashLSH).

    Política de resolução (auditoria §37): a comparação é cega a rótulos e,
    em cada grupo de quase-duplicatas, sobrevive a primeira ocorrência na
    ordem do DataFrame (menor índice) — um par com rótulos distintos tem o
    membro de maior índice removido, sem regra ligada à classe.

    Args:
        df: DataFrame com coluna de texto.
        text_col: nome da coluna com texto a comparar.
        thr: limiar Jaccard pra near-duplicate (0.85 = conservador).
        num_perm: nº de permutações do MinHash (128 = bom trade-off).
        shingle: tamanho dos shingles de caracteres (5).

    Returns:
        (df_clean, n_duplicatas_exatas, n_near_duplicates_removidos)
    """
    df = df.copy()
    df["_clean"] = df[text_col].astype(str).str.strip().str.lower()
    df["_sha1"] = df["_clean"].apply(
        lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()
    )
    n_exact = int(df["_sha1"].duplicated().sum())
    df = df.drop_duplicates("_sha1").reset_index(drop=True)

    lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
    mhs: dict[int, MinHash] = {}
    for i, txt in enumerate(tqdm(df["_clean"], desc="MinHash", leave=False)):
        m = MinHash(num_perm=num_perm)
        for sh in set(txt[j : j + shingle] for j in range(max(0, len(txt) - shingle + 1))):
            m.update(sh.encode("utf-8"))
        mhs[i] = m
        lsh.insert(f"d_{i}", m)

    to_drop: set[int] = set()
    for i, m in mhs.items():
        for r in lsh.query(m):
            o = int(r.split("_")[1])
            if o != i and o not in to_drop and i not in to_drop:
                to_drop.add(o)

    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    df = df.drop(columns=["_clean", "_sha1"])
    return df, n_exact, len(to_drop)


def load_and_prepare(
    variant: str,
    encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder, list[str]]:
    """Conveniência: load_fakerecogna → normalize_schema → encode_labels → parse_dates → dedupe.

    Args:
        variant: 'abstrativa' ou 'extrativa'.
        encoder: opcional, reusa LabelEncoder pra manter mapping entre variantes.

    Returns:
        (df_limpo_dedupado, encoder, class_names)
    """
    df = load_fakerecogna(variant)
    df = normalize_schema(df)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias faltando: {missing}")
    df, encoder, class_names = encode_labels(df, encoder)
    df = parse_dates(df)
    df, n_ex, n_nr = dedupe(df, text_col="text")
    log.info(f"{variant}: exatas={n_ex}, near={n_nr}, final={len(df)}")
    return df, encoder, class_names


__all__ = [
    "load_fakerecogna",
    "normalize_schema",
    "encode_labels",
    "parse_dates",
    "dedupe",
    "load_and_prepare",
]
