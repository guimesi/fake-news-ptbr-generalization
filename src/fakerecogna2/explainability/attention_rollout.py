"""Attention Rollout — Abnar & Zuidema 2020 (Seção 18.2, cell 66)."""

from __future__ import annotations


import numpy as np

from ..config import BERTIMBAU_MODEL, MAX_LEN




# -- API limpa ----------------------------------------------------------------
class AttentionRollout:
    """Carrega uma cópia 'eager' do BERTimbau pra extrair attentions.

    Reusa pesos de um `bert_clf` fine-tuned (ou os pesos pretrained do hub
    se você passar `state_dict=None`).
    """

    def __init__(
        self,
        bert_clf,
        tokenizer,
        device: str = "cpu",
        model_name: str = BERTIMBAU_MODEL,
        max_seq_len: int = MAX_LEN,
    ):
        from transformers import AutoModel

        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self._eager = (
            AutoModel.from_pretrained(model_name, attn_implementation="eager")
            .to(device)
            .eval()
        )
        if bert_clf is not None:
            self._eager.load_state_dict(bert_clf.bert.state_dict())

    def explain(
        self, text: str, discard_ratio: float = 0.9
    ) -> tuple[list[str], np.ndarray]:
        """Rollout: percorre todas as camadas com identidade + normalização por linha.

        Returns:
            (tokens_validos, scores) — scores são a linha do [CLS] da matriz final.
        """
        import torch

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self._eager(
                enc["input_ids"], enc["attention_mask"], output_attentions=True
            )
        attentions = out.attentions  # tupla de (B, H, S, S)

        result = torch.eye(attentions[0].size(-1)).to(self.device)
        for attn in attentions:
            attn_mean = attn.mean(dim=1).squeeze(0)
            threshold = torch.quantile(attn_mean.view(-1), discard_ratio)
            attn_mean = torch.where(
                attn_mean < threshold, torch.zeros_like(attn_mean), attn_mean
            )
            attn_mean = attn_mean + torch.eye(attn_mean.size(-1)).to(self.device)
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            result = attn_mean @ result

        cls_attention = result[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())

        valid = [
            (t, a) for t, a in zip(tokens, cls_attention)
            if t not in ("[PAD]", "[CLS]", "[SEP]")
        ]
        if not valid:
            return [], np.array([])
        toks, sc = zip(*valid)
        return list(toks), np.array(sc)


__all__ = [
    "AttentionRollout",
]
