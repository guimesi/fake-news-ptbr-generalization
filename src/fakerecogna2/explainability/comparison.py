"""Comparação visual IG × Attention Rollout (Seção 18.3, cell 67)."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..config import MAX_LEN
from ..utils.io_utils import save_plot
from .attention_rollout import AttentionRollout
from .integrated_gradients import compute_integrated_gradients


def _bert_predict(bert_clf, tokenizer, text: str, device: str) -> int:
    """Predição do próprio BERTimbau FT (classe-alvo correta para o IG)."""
    import torch

    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = bert_clf(enc["input_ids"], enc["attention_mask"])
    return int(logits.argmax(-1).item())


# -- API limpa ----------------------------------------------------------------
def compare_xai_for_examples(
    X_test: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    bert_clf,
    tokenizer,
    targets: list[tuple[int, str]],
    device: str = "cpu",
    n_examples: int = 12,
    top_k: int = 20,
    save_dir: str = "lime",
) -> None:
    """Plota IG e Attention Rollout lado a lado pros mesmos exemplos LIME.

    `predictions` (predições do modelo que selecionou os alvos, ex.: Ens3)
    é mantido por compatibilidade de assinatura, mas a classe-alvo do IG é
    a predição do próprio BERTimbau FT (ver _bert_predict).
    """
    rollout = AttentionRollout(bert_clf, tokenizer, device=device)

    for si, (idx, tag) in enumerate(tqdm(targets[:n_examples], desc="XAI compare")):
        text = X_test[idx]
        true_cls = int(y_test[idx])
        # Classe-alvo do IG = predição do PRÓPRIO BERTimbau FT (não a do
        # modelo que selecionou os alvos, ex.: Ens3) — correção da auditoria
        # de 2026-08; inócuo em células de acerto, relevante para FP/FN.
        pred_cls = _bert_predict(bert_clf, tokenizer, text, device)

        ig_tokens, ig_scores = compute_integrated_gradients(
            text, pred_cls, bert_clf, tokenizer, device=device
        )
        ar_tokens, ar_scores = rollout.explain(text)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, (toks, sc, title) in zip(
            axes,
            [
                (ig_tokens[:40], ig_scores[:40], "Integrated Gradients"),
                (ar_tokens[:40], ar_scores[:40], "Attention Rollout"),
            ],
        ):
            if len(toks) == 0:
                continue
            order = np.argsort(np.abs(sc))[::-1][:top_k]
            toks_ = [toks[i] for i in order]
            sc_ = sc[order]
            colors = ["#2E7D32" if s > 0 else "#C62828" for s in sc_]
            ax.barh(range(len(toks_)), sc_, color=colors, alpha=0.85)
            ax.set_yticks(range(len(toks_)))
            ax.set_yticklabels(toks_)
            ax.set_title(
                f"{title} — {tag}\nTrue={class_names[true_cls]} Pred(BERT)={class_names[pred_cls]}"
            )
            ax.axvline(0, color="k", lw=0.5)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_plot(fig, f"{save_dir}/xai_gradient_{si:02d}_{tag}")


__all__ = [
    "compare_xai_for_examples",
]
