"""Integrated Gradients via Captum (Seção 18.1, cell 65)."""

from __future__ import annotations


import numpy as np

from ..config import MAX_LEN
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def compute_integrated_gradients(
    text: str,
    target_class: int,
    bert_clf,
    tokenizer,
    device: str = "cpu",
    max_seq_len: int = MAX_LEN,
    n_steps: int = 50,
) -> tuple[list[str], np.ndarray]:
    """Integrated Gradients na camada de embeddings do BERTimbau FT.

    Args:
        text: texto a explicar.
        target_class: classe alvo.
        bert_clf: classifier BERTimbau fine-tuned (com .bert.embeddings).
        tokenizer: tokenizer compatível.
        device, max_seq_len, n_steps: hiperparâmetros padrão.

    Returns:
        (tokens_validos, scores) ou ([], []) se captum indisponível.
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        log.warning("captum não instalado — pulando Integrated Gradients.")
        return [], np.array([])

    import torch

    bert_clf.eval()
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    ).to(device)
    input_ids = enc["input_ids"]
    mask = enc["attention_mask"]
    pad_id = tokenizer.pad_token_id
    baseline_ids = torch.full_like(input_ids, pad_id)

    lig = LayerIntegratedGradients(
        lambda ids, m: bert_clf(ids, m),
        bert_clf.bert.embeddings,
    )

    attrs, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(mask,),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    valid = [
        (t, a) for t, a in zip(tokens, attrs) if t not in ("[PAD]", "[CLS]", "[SEP]")
    ]
    if not valid:
        return [], np.array([])
    toks, sc = zip(*valid)
    return list(toks), np.array(sc)


__all__ = [
    "compute_integrated_gradients",
]
