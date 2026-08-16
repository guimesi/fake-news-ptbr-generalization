"""Estabilidade LIME: K execuções + Jaccard sobre top-N tokens (Seção I, cell 112)."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import ARTIFACTS_DIR, SEED
from ..utils.logging_utils import get_logger
from .lime_explainer import ProbaPredictor

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def lime_stability(
    X_test: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    predict_fn: ProbaPredictor,
    class_names: list[str],
    n_correct: int = 5,
    n_wrong: int = 5,
    k_runs: int = 5,
    top_n: int = 10,
    num_samples: int = 300,
    save_as: str | None = "I_lime_stability",
    seed: int = SEED,
) -> pd.DataFrame:
    """Roda LIME K vezes em N exemplos e calcula Jaccard entre runs.

    Returns:
        DataFrame com idx, true, pred, correct, jaccard_mean/min/max + linha 'MÉDIA'.
    """
    import lime
    import lime.lime_text

    explainer = lime.lime_text.LimeTextExplainer(
        class_names=class_names, random_state=None
    )

    rng = np.random.RandomState(seed)
    correct_idx = np.where(predictions == y_test)[0]
    wrong_idx = np.where(predictions != y_test)[0]
    sample_idx = np.concatenate(
        [
            rng.choice(correct_idx, min(n_correct, len(correct_idx)), replace=False),
            rng.choice(wrong_idx, min(n_wrong, len(wrong_idx)), replace=False),
        ]
    )

    log.info(
        f"[Estabilidade LIME] {len(sample_idx)} exemplos × {k_runs} runs "
        f"= {len(sample_idx)*k_runs} execuções LIME."
    )

    rows: list[dict] = []
    for idx in tqdm(sample_idx, desc="LIME stability"):
        text = X_test[int(idx)]
        tops: list[set[str]] = []
        for _ in range(k_runs):
            exp = explainer.explain_instance(
                text, predict_fn, num_features=top_n, num_samples=num_samples
            )
            tops.append({f[0] for f in exp.as_list()[:top_n]})
        jaccards = [_jaccard(a, b) for a, b in combinations(tops, 2)]
        rows.append(
            {
                "idx": int(idx),
                "true": int(y_test[idx]),
                "pred": int(predictions[idx]),
                "correct": bool(y_test[idx] == predictions[idx]),
                "jaccard_mean": float(np.mean(jaccards)) if jaccards else 1.0,
                "jaccard_min": float(min(jaccards)) if jaccards else 1.0,
                "jaccard_max": float(max(jaccards)) if jaccards else 1.0,
            }
        )

    df = pd.DataFrame(rows).round(4)
    mean_stability = float(df["jaccard_mean"].mean())
    df.loc[len(df)] = {
        "idx": "MÉDIA",
        "true": -1,
        "pred": -1,
        "correct": True,
        "jaccard_mean": mean_stability,
        "jaccard_min": float(df["jaccard_min"].mean()),
        "jaccard_max": float(df["jaccard_max"].mean()),
    }

    print(
        "\n=== Estabilidade LIME (Jaccard sobre top-N tokens, K execuções) ==="
    )
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

    if mean_stability >= 0.7:
        msg = "🟢 LIME ESTÁVEL — reportar com confiança."
    elif mean_stability >= 0.4:
        msg = "🟡 LIME PARCIALMENTE ESTÁVEL — reportar com cautela."
    else:
        msg = "🔴 LIME INSTÁVEL — triangular com IG/Rollout."
    log.info(f"Jaccard médio: {mean_stability:.3f}  {msg}")

    if save_as is not None:
        out_path = ARTIFACTS_DIR / "metrics" / f"{save_as}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(df.to_csv(index=False), encoding="utf-8")

    return df


__all__ = [
    "lime_stability",
]
