"""Paraphrasing Equalizer — TextRank + dataset equalizado.

Cobre as células 76, 77, 78 (Seção 21). A célula 78 (ensemble) virou
parte do orquestrador `scripts/11_paraphrasing_equalizer.py` (chama
preprocessing aqui + features + models + evaluation).
"""

from __future__ import annotations


import pandas as pd

from ..utils.logging_utils import get_logger
from .text_cleaning import preprocess_base

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def _has_sumy() -> bool:
    try:
        import sumy  # noqa: F401
        return True
    except ImportError:
        return False


def extractive_summarize(text: str, n_sentences: int = 4) -> str:
    """Sumariza via TextRank (Mihalcea & Tarau 2004).

    Determinístico, ~10ms/texto em CPU. Retorna o texto original se sumy
    não estiver instalado ou em caso de erro.
    """
    if not _has_sumy():
        return text
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer

    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer("portuguese"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, n_sentences)
        return " ".join(str(s) for s in summary) or text
    except Exception:
        return text


def estimate_n_sentences(
    df: pd.DataFrame, label_col: str = "label_enc", text_col: str = "text"
) -> int:
    """Estima n_sentences para equalizar comprimento médio das classes.

    Toma o comprimento médio das **reais** (label_enc == 1) e divide por 20
    (≈ tokens por sentença em PT-BR). Mínimo 2.
    """
    avg_len_real = df[df[label_col] == 1][text_col].str.split().str.len().mean()
    avg_len_fake = df[df[label_col] == 0][text_col].str.split().str.len().mean()
    log.info(f"Média tokens — real: {avg_len_real:.0f}, fake: {avg_len_fake:.0f}")
    n_sent = max(2, int(avg_len_real / 20))
    log.info(f"Usando n_sentences={n_sent} para equalizar comprimento.")
    return n_sent


def build_equalized_dataset(
    df: pd.DataFrame,
    n_sentences: int | None = None,
    label_col: str = "label_enc",
    text_col: str = "text",
    out_col: str = "text_eq",
    out_proc_col: str = "text_eq_proc",
    fake_label: int = 0,
    min_length: int = 10,
) -> pd.DataFrame:
    """Sumariza o lado fake e re-aplica preprocess_base para versão equalizada.

    Retorna df com colunas extras `out_col` (texto sumarizado) e
    `out_proc_col` (texto sumarizado + pré-processado), filtrando textos
    curtos.
    """
    from tqdm.auto import tqdm

    if n_sentences is None:
        n_sentences = estimate_n_sentences(df, label_col=label_col, text_col=text_col)

    out = df.copy()
    fake_mask = out[label_col] == fake_label
    tqdm.pandas(desc="Sumarizando fakes")
    out.loc[fake_mask, out_col] = out.loc[fake_mask, text_col].progress_apply(
        lambda t: extractive_summarize(str(t), n_sentences)
    )
    out.loc[~fake_mask, out_col] = out.loc[~fake_mask, text_col]
    out[out_proc_col] = out[out_col].apply(preprocess_base)
    out = out[out[out_proc_col].str.len() > min_length].reset_index(drop=True)

    lens_fake = out[out[label_col] == fake_label][out_col].str.split().str.len().mean()
    lens_real = out[out[label_col] != fake_label][out_col].str.split().str.len().mean()
    log.info(
        f"Após equalização — real: {lens_real:.0f} tokens, fake: {lens_fake:.0f} tokens"
    )
    return out


__all__ = [
    "extractive_summarize",
    "estimate_n_sentences",
    "build_equalized_dataset",
]
