"""LIME para acertos e erros (Seção 15.1, cell 50)."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import LIME_NUM_SAMPLES, SEED
from ..utils.io_utils import save_plot, save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
ProbaPredictor = Callable[[list[str]], np.ndarray]


def select_lime_targets(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    n_per_bucket: int = 2,
    seed: int = SEED,
) -> list[tuple[int, str]]:
    """Seleciona acertos (TP) e erros (FN) balanceados por classe.

    Returns:
        Lista de (idx, tag) onde tag é 'TP_<classe>' ou 'FN_<classe>'.
    """
    rng = np.random.RandomState(seed)
    targets: list[tuple[int, str]] = []
    for ci, cn in enumerate(class_names):
        idxs = np.where((y_test == ci) & (predictions == ci))[0]
        if len(idxs) > 0:
            for i in rng.choice(idxs, min(n_per_bucket, len(idxs)), replace=False):
                targets.append((int(i), f"TP_{cn}"))
    for ci, cn in enumerate(class_names):
        idxs = np.where((y_test == ci) & (predictions != ci))[0]
        if len(idxs) > 0:
            for i in rng.choice(idxs, min(n_per_bucket, len(idxs)), replace=False):
                targets.append((int(i), f"FN_{cn}"))
    return targets


def select_lime_targets_4cells(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    positive_class_idx: int = 0,
    n_per_bucket: int = 3,
    seed: int = SEED,
) -> list[tuple[int, str]]:
    """Seleciona exemplos das 4 celulas da matriz de confusao binaria.

    Atende a declaracao da Secao 5.10 do Cap. 5: ampliar XAI para
    TP/TN/FP/FN.

    Args:
        positive_class_idx: classe tabulada como positiva nas celulas
            TP/TN/FP/FN (default 0). Convencao canonica do corpus:
            0 = verdadeira, 1 = fake (CHANGELOG §14).
        n_per_bucket: quantos exemplos amostrar por celula (4 buckets totais).

    Returns:
        Lista de (idx, tag) onde tag e 'TP'/'TN'/'FP'/'FN'.
    """
    pos = positive_class_idx
    neg = 1 - pos
    rng = np.random.RandomState(seed)
    buckets = {
        "TP": np.where((y_test == pos) & (predictions == pos))[0],
        "TN": np.where((y_test == neg) & (predictions == neg))[0],
        "FP": np.where((y_test == neg) & (predictions == pos))[0],
        "FN": np.where((y_test == pos) & (predictions == neg))[0],
    }
    targets: list[tuple[int, str]] = []
    for tag, idxs in buckets.items():
        if len(idxs) > 0:
            sample = rng.choice(idxs, min(n_per_bucket, len(idxs)), replace=False)
            for i in sample:
                targets.append((int(i), tag))
    return targets


def _parse_lime_features(feats_str: str) -> list[tuple[str, float]]:
    """Parseia 'features' de LIME records, tolerando np.str_/np.float64 reprs.

    Strings podem vir como `[(np.str_('tok'), np.float64(-0.123)), ...]` ou
    como `[('tok', -0.123), ...]`. Usa regex para extrair pares.
    """
    import re
    # Padrao: ('token', valor) OU (np.str_('token'), np.float64(valor))
    pattern = re.compile(
        r"\(\s*(?:np\.str_\()?\s*['\"]([^'\"]+)['\"]\)?\s*,\s*"
        r"(?:np\.\w+\()?\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)?\s*\)"
    )
    out: list[tuple[str, float]] = []
    for tok, val in pattern.findall(feats_str):
        try:
            out.append((tok, float(val)))
        except ValueError:
            continue
    return out


def aggregate_lime_tokens_by_tag(
    records_df: pd.DataFrame,
    top_n: int = 20,
    save_as: str | None = "15_lime_tokens_by_tag",
) -> pd.DataFrame:
    """Agrega tokens mais frequentes/contributivos por tag (TP/TN/FP/FN).

    A coluna `features` do records_df vem como str(list[(token, weight)]).
    Reconstroi a lista e soma os pesos absolutos por token, dentro de
    cada tag.

    Returns:
        DataFrame longo com colunas (tag, token, abs_weight_sum, n_exemplos).
    """
    from collections import defaultdict
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for _, row in records_df.iterrows():
        tag = row["tag"]
        feats = _parse_lime_features(str(row["features"]))
        if not feats:
            continue
        counts[tag] += 1
        for tok, w in feats:
            agg[tag][tok] += abs(float(w))

    rows = []
    for tag, tok_w in agg.items():
        top = sorted(tok_w.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for tok, w in top:
            rows.append({
                "tag": tag,
                "token": tok,
                "abs_weight_sum": round(w, 4),
                "n_examples": counts[tag],
            })
    df = pd.DataFrame(rows)
    if save_as is not None:
        save_table(df, save_as)
    return df


def run_lime_explanations(
    X_test: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    predict_fn: ProbaPredictor,
    class_names: list[str],
    targets: list[tuple[int, str]] | None = None,
    num_features: int = 15,
    num_samples: int = LIME_NUM_SAMPLES,
    save_dir: str = "lime",
    save_as: str | None = "15_lime_records",
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Roda LIME em uma lista de exemplos e salva os plots.

    Returns:
        (df_records, targets_usados)
    """
    import lime
    import lime.lime_text

    if targets is None:
        targets = select_lime_targets(y_test, predictions, class_names)

    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names, random_state=SEED)
    log.info(f"LIME em {len(targets)} exemplos (acertos + erros)")

    records: list[dict] = []
    for si, (idx, tag) in enumerate(tqdm(targets, desc="LIME")):
        exp = explainer.explain_instance(
            X_test[idx], predict_fn, num_features=num_features, num_samples=num_samples
        )
        feats = exp.as_list()
        ws = [f[0] for f in feats]
        wts = [f[1] for f in feats]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            range(len(ws)),
            wts,
            color=["#2E7D32" if w > 0 else "#C62828" for w in wts],
            alpha=0.85,
        )
        ax.set_yticks(range(len(ws)))
        ax.set_yticklabels(ws)
        ax.set_xlabel("Contribuição")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(
            f"LIME — {tag} (idx={idx}, true={class_names[int(y_test[idx])]}, "
            f"pred={class_names[int(predictions[idx])]})"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_plot(fig, f"{save_dir}/lime_{si:02d}_{tag}")
        records.append({"idx": int(idx), "tag": tag, "features": str(feats)})

    df = pd.DataFrame(records)
    if save_as is not None:
        save_table(df, save_as)
    return df, targets


__all__ = [
    "ProbaPredictor",
    "select_lime_targets",
    "select_lime_targets_4cells",
    "aggregate_lime_tokens_by_tag",
    "run_lime_explanations",
]
