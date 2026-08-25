"""Back-translation PT→EN→PT via MarianMT (Seção 19.2, cell 70).

NOTA (auditoria §37): os checkpoints usados são os multilíngues românicos
`Helsinki-NLP/opus-mt-roa-en` / `opus-mt-en-roa` (com tag de idioma-alvo
``>>por<<``), e NÃO os pares dedicados pt-en; decodificação com num_beams=2,
sem cache de traduções, 1 texto por chamada.

Uso típico::

    bt = BackTranslator(device="cuda")
    perturbed_texts = [bt(t) for t in texts]
    bt.unload()  # libera VRAM
"""

from __future__ import annotations


from ..config import BT_EN_TO_PT_MODEL, BT_PT_TO_EN_MODEL
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
class BackTranslator:
    """Encapsula os dois modelos MarianMT (PT→EN e EN→PT)."""

    PT_EN_MODEL = BT_PT_TO_EN_MODEL
    EN_PT_MODEL = BT_EN_TO_PT_MODEL

    def __init__(self, device: str = "cpu") -> None:
        from transformers import MarianMTModel, MarianTokenizer

        log.info(f"Carregando modelos de back-translation em {device}...")
        self.device = device
        self.pt_en_tok = MarianTokenizer.from_pretrained(self.PT_EN_MODEL)
        self.pt_en_model = MarianMTModel.from_pretrained(self.PT_EN_MODEL).to(device).eval()
        self.en_pt_tok = MarianTokenizer.from_pretrained(self.EN_PT_MODEL)
        self.en_pt_model = MarianMTModel.from_pretrained(self.EN_PT_MODEL).to(device).eval()

    def translate(self, text: str, max_tokens: int = 512) -> str:
        """PT → EN → PT. Trunca a 2000 chars de entrada e max_tokens por direção."""
        import torch

        with torch.no_grad():
            # PT → EN
            batch = self.pt_en_tok(
                [text[:2000]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(self.device)
            trans_en = self.pt_en_model.generate(
                **batch, max_new_tokens=max_tokens, num_beams=2
            )
            en_text = self.pt_en_tok.decode(trans_en[0], skip_special_tokens=True)

            # EN → PT (tag de idioma alvo)
            en_text_tagged = ">>por<< " + en_text
            batch = self.en_pt_tok(
                [en_text_tagged],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(self.device)
            trans_pt = self.en_pt_model.generate(
                **batch, max_new_tokens=max_tokens, num_beams=2
            )
            return self.en_pt_tok.decode(trans_pt[0], skip_special_tokens=True)

    def __call__(self, text: str, max_tokens: int = 512) -> str:
        return self.translate(text, max_tokens)

    def unload(self) -> None:
        """Libera os modelos da memória (importante após uso na GPU)."""
        import gc

        import torch

        del self.pt_en_model, self.en_pt_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "BackTranslator",
]
