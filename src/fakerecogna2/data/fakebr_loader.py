"""Carregador local do Fake.br-Corpus pra avaliação cross-dataset.

Cobre a célula 57 do notebook (Seção 17.1). Download:
https://github.com/roneysco/Fake.br-Corpus
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import FAKEBR_LOCAL_PATH
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def load_fakebr_corpus(
    base_path: str | Path = FAKEBR_LOCAL_PATH,
    version: str = "size_normalized_texts",
) -> pd.DataFrame:
    """Carrega o Fake.br-Corpus do diretório local.

    Args:
        base_path: caminho da pasta raiz `Fake.br-Corpus-master`. Padrão é
            o `FAKEBR_LOCAL_PATH` do `configs/config.yaml`.
        version: subpasta a usar:
            - 'size_normalized_texts' (recomendado): real truncada ao tamanho
              da fake pareada (sem viés de comprimento).
            - 'full_texts': textos completos originais.
            - 'preprocessed': CSV único pré-processado (sem stopwords/acentos).

    Returns:
        DataFrame com colunas: text, label ('fake'/'real'), source_file e
        (quando disponível) author, link, category, date.

    Raises:
        FileNotFoundError: se o diretório base ou a versão pedida não existir.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(
            f"Pasta {base} não encontrada. Baixe o Fake.br-Corpus de "
            "https://github.com/roneysco/Fake.br-Corpus e extraia no caminho indicado."
        )

    if version == "preprocessed":
        return _load_preprocessed_csv(base)

    return _load_text_files(base, version)


def _load_preprocessed_csv(base: Path) -> pd.DataFrame:
    csv_candidates = list((base / "preprocessed").glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {base/'preprocessed'}")
    df = pd.read_csv(csv_candidates[0])
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("preprocessed_news", "text", "texto"):
            col_map[c] = "text"
        elif cl == "label":
            col_map[c] = "label"
    df = df.rename(columns=col_map)
    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .map({"0": "fake", "1": "real", "fake": "fake", "true": "real"})
        .fillna(df["label"])
    )
    log.info(f"Fake.br-Corpus (preprocessed CSV): {df.shape}")
    return df[["text", "label"]].dropna()


def _load_text_files(base: Path, version: str) -> pd.DataFrame:
    ver_dir = base / version
    if not ver_dir.exists():
        raise FileNotFoundError(f"Diretório {ver_dir} não encontrado.")

    rows: list[dict] = []
    for label_dir, label_name in [("fake", "fake"), ("true", "real")]:
        d = ver_dir / label_dir
        if not d.exists():
            log.warning(f"Subpasta {d} não encontrada — pulando.")
            continue
        files = sorted(
            d.glob("*.txt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
        )
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                txt = f.read_text(encoding="latin-1", errors="ignore").strip()
            if not txt:
                continue

            meta: dict = {}
            if version == "full_texts":
                meta_dir = ver_dir / f"{label_dir}-meta-information"
                meta_file = meta_dir / f.name if meta_dir.exists() else None
                if meta_file and meta_file.exists():
                    try:
                        lines = meta_file.read_text(
                            encoding="utf-8", errors="ignore"
                        ).split("\n")
                        if len(lines) >= 4:
                            meta = {
                                "author": lines[0].strip(),
                                "link": lines[1].strip(),
                                "category": lines[2].strip(),
                                "date": lines[3].strip(),
                            }
                    except Exception:
                        pass

            rows.append({"text": txt, "label": label_name, "source_file": f.name, **meta})

    df = pd.DataFrame(rows)
    log.info(
        f"Fake.br-Corpus ({version}): {df.shape} "
        f'— fake={(df.label=="fake").sum()} real={(df.label=="real").sum()}'
    )
    return df


__all__ = [
    "load_fakebr_corpus",
]
