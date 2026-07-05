"""Comparação visual IG × Attention Rollout."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils.io_utils import save_plot
from .attention_rollout import AttentionRollout
from .integrated_gradients import compute_integrated_gradients




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
    """Plota IG e Attention Rollout lado a lado pros mesmos exemplos LIME."""
    rollout = AttentionRollout(bert_clf, tokenizer, device=device)

    for si, (idx, tag) in enumerate(tqdm(targets[:n_examples], desc="XAI compare")):
        text = X_test[idx]
        true_cls = int(y_test[idx])
        pred_cls = int(predictions[idx])

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
                f"{title}: {tag}\nTrue={class_names[true_cls]} Pred={class_names[pred_cls]}"
            )
            ax.axvline(0, color="k", lw=0.5)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_plot(fig, f"{save_dir}/xai_gradient_{si:02d}_{tag}")


__all__ = [
    "compare_xai_for_examples",
]
