"""Extração de embeddings BERTimbau (Seção 7, cells 25 e 26).

Encapsula o backbone (tokenizer + AutoModel) num `EmbeddingExtractor` que
expõe `extract_token_embs` e `extract_cls_embs`. Lazy import de torch/transformers.
"""

from __future__ import annotations


from ..config import BERTIMBAU_MODEL, MAX_LEN
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
class EmbeddingExtractor:
    """Encapsula tokenizer + BERTimbau backbone pra extrair embeddings."""

    def __init__(
        self,
        model_name: str = BERTIMBAU_MODEL,
        max_seq_len: int = MAX_LEN,
        device: str = "cpu",
    ):
        from transformers import AutoModel, AutoTokenizer

        log.info(f"Carregando {model_name} em {device}...")
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.embed_dim = self.model.config.hidden_size
        log.info(f"EMBED_DIM={self.embed_dim}")

    def _extract(
        self, texts, mode: str = "token", max_len: int | None = None, batch_size: int = 32
    ):
        import torch
        from tqdm.auto import tqdm

        max_len = max_len if max_len is not None else self.max_seq_len
        outs = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size), desc=f"BERT [{mode}]", leave=False
            ):
                enc = self.tokenizer(
                    list(texts[i : i + batch_size]),
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**enc).last_hidden_state
                outs.append(h.cpu() if mode == "token" else h[:, 0, :].cpu())
        return torch.cat(outs, 0)

    def extract_token_embs(self, texts, **kw):
        """Embeddings token-level (shape: [N, max_len, hidden_size])."""
        return self._extract(texts, mode="token", **kw)

    def extract_cls_embs(self, texts, **kw):
        """[CLS] embeddings (shape: [N, hidden_size])."""
        return self._extract(texts, mode="cls", **kw)

    def unload(self) -> None:
        """Libera o backbone da memória/GPU."""
        import gc

        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_all_embeddings(
    extractor: EmbeddingExtractor,
    X_train: list[str],
    X_val: list[str],
    X_test: list[str],
) -> dict[str, "torch.Tensor"]:
    """Atalho: extrai token e CLS pros 3 splits e devolve dict."""
    log.info("Extraindo token embeddings...")
    emb_train = extractor.extract_token_embs(X_train)
    emb_val = extractor.extract_token_embs(X_val)
    emb_test = extractor.extract_token_embs(X_test)
    log.info("Extraindo [CLS] embeddings...")
    cls_train = extractor.extract_cls_embs(X_train)
    cls_val = extractor.extract_cls_embs(X_val)
    cls_test = extractor.extract_cls_embs(X_test)
    log.info(f"Shapes — tok: {emb_train.shape}  cls: {cls_train.shape}")
    return {
        "token_train": emb_train,
        "token_val": emb_val,
        "token_test": emb_test,
        "cls_train": cls_train,
        "cls_val": cls_val,
        "cls_test": cls_test,
    }


__all__ = [
    "EmbeddingExtractor",
    "extract_all_embeddings",
]
