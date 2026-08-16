"""Equalizador linguístico — remoção de metatexto fact-check (Seção F).

Cobre as células 104 e 105. A célula 105 (ensemble + comparativo) virou
parte do orquestrador `scripts/11_paraphrasing_equalizer.py`.
"""

from __future__ import annotations


import pandas as pd

from ..utils.logging_utils import get_logger
from .text_cleaning import preprocess_base

log = get_logger()


METATEXT_FAKECHECK: set[str] = {
    # Verbos de circulação/disseminação
    "circula", "circulam", "circulou", "circulando",
    "viraliza", "viralizou", "viralizada", "viralizado",
    "compartilha", "compartilhada", "compartilhado", "compartilhando", "compartilhou",
    "repassa", "repassada", "repassado", "repassando", "repassou",
    # Plataformas/redes
    "facebook", "whatsapp", "twitter", "instagram", "youtube", "telegram", "tiktok",
    "redes", "sociais", "rede", "social", "plataforma", "plataformas",
    # Formatos de mídia viral
    "vídeo", "vídeos", "video", "videos",
    "foto", "fotos", "imagem", "imagens",
    "publicação", "publicações", "publicou", "publicada", "publicado",
    "postagem", "postagens", "postou", "postada", "postado",
    "mensagem", "mensagens", "áudio", "áudios",
    # Verbos de atribuição típicos de fact-check
    "mostra", "mostram", "mostraria",
    "afirma", "afirmam", "afirmaria", "afirmou",
    "alega", "alegam", "alegou",
    "diz", "dizem", "dizia", "dizendo",
    # Modificadores de veiculação
    "aqui", "leia", "vejam", "veja", "assista", "assistam", "confira", "confiram",
    "fake", "boato", "rumor", "desinformação", "desinformacao",
}




# -- API limpa ----------------------------------------------------------------
def strip_metatext(text: str, stop_list: set[str] = METATEXT_FAKECHECK) -> str:
    """Remove tokens metatextuais da versão preprocessada do texto."""
    tokens = preprocess_base(text).split()
    return " ".join(t for t in tokens if t not in stop_list)


def build_nometatext_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "text_nometa",
    stop_list: set[str] = METATEXT_FAKECHECK,
    min_length: int = 10,
) -> pd.DataFrame:
    """Aplica `strip_metatext` em todo o df e reporta % de tokens removidos."""
    from tqdm.auto import tqdm

    tqdm.pandas(desc="Strip metatext")
    out = df.copy()
    out[out_col] = out[text_col].progress_apply(lambda t: strip_metatext(t, stop_list))
    out = out[out[out_col].str.len() > min_length].reset_index(drop=True)

    orig_len = df[text_col].apply(lambda x: len(preprocess_base(x).split()))
    new_len = out[out_col].apply(lambda x: len(x.split()))
    removed_pct = 1 - (new_len.mean() / orig_len.mean())
    log.info(
        f"[Equalizador Linguístico] {len(stop_list)} tokens, "
        f"removidos em média: {removed_pct*100:.1f}% das palavras."
    )
    return out


__all__ = [
    "METATEXT_FAKECHECK",
    "strip_metatext",
    "build_nometatext_dataset",
]
