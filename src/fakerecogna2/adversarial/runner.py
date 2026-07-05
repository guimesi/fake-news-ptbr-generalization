"""Orquestrador dos experimentos adversariais.

Em vez de assumir `predict_ens3` / `predict_bert_ft` como globais, recebe um dict `models` mapeando nome → função `(list[str]) -> np.ndarray`
de probabilidades. Isso desacopla este módulo dos detalhes de models/.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import (
    ADV_BT_SAMPLE_SIZE,
    ADV_DELETION_RATE,
    ADV_PERTURB_SAMPLE_SIZE,
    ADV_SWAP_RATE,
    ADV_TYPO_RATE,
    SEED,
)
from ..utils.io_utils import save_table
from ..utils.logging_utils import get_logger
from .back_translation import BackTranslator
from .perturbations import adv_random_typos, adv_word_deletion, adv_word_swap

log = get_logger()




# -- API limpa ----------------------------------------------------------------
ProbaPredictor = Callable[[list[str]], np.ndarray]


def run_adversarial_eval(
    X_test: list[str],
    y_test: np.ndarray,
    models: dict[str, ProbaPredictor],
    n_sample: int = ADV_PERTURB_SAMPLE_SIZE,
    n_bt_sample: int = ADV_BT_SAMPLE_SIZE,
    do_back_translation: bool = True,
    device: str = "cpu",
    seed: int = SEED,
    save_as: str | None = "19_adversarial_robustness",
) -> pd.DataFrame:
    """Avalia robustez sob 4 perturbações: typos, deletion, swap, back-translation.

    Args:
        X_test: textos do teste.
        y_test: rótulos do teste.
        models: dict nome→função que devolve probabilidades (shape [N, num_classes]).
        n_sample: subsample do teste (mantém custo razoável).
        n_bt_sample: subsample menor pra back-translation (caríssimo).
        do_back_translation: liga/desliga MarianMT.
        device: 'cpu' ou 'cuda' (passado pro BackTranslator se ligado).
        seed: semente p/ rng das perturbações.
        save_as: nome do CSV em outputs/metrics/ (None pra não salvar).

    Returns:
        DataFrame: Model, Perturbation, N, F1_orig, F1_pert, flip_rate, ΔF1.
    """
    n = min(n_sample, len(X_test))
    rng = random.Random(seed)
    idx = rng.sample(range(len(X_test)), n)
    X_orig = [X_test[i] for i in idx]
    y_arr = np.asarray(y_test)
    y = y_arr[idx]

    log.info(f"Adversarial: n={n} (BT={n_bt_sample if do_back_translation else 0})")
    orig_preds = {name: fn(X_orig).argmax(1) for name, fn in models.items()}

    perturbations: dict[str, Callable[[list[str]], list[str]] | None] = {
        f"Typos ({ADV_TYPO_RATE:.0%})": lambda txts: [
            adv_random_typos(t, ADV_TYPO_RATE, rng) for t in txts
        ],
        f"Word deletion ({ADV_DELETION_RATE:.0%})": lambda txts: [
            adv_word_deletion(t, ADV_DELETION_RATE, rng) for t in txts
        ],
        f"Word swap ({ADV_SWAP_RATE:.0%})": lambda txts: [
            adv_word_swap(t, ADV_SWAP_RATE, rng) for t in txts
        ],
    }
    if do_back_translation:
        perturbations[f"Back-translation ({n_bt_sample} sample)"] = None  # tratado abaixo

    rows: list[dict] = []
    bt: BackTranslator | None = None
    try:
        for pert_name, fn in perturbations.items():
            if pert_name.startswith("Back-translation"):
                if bt is None:
                    bt = BackTranslator(device=device)
                idx_bt = idx[:n_bt_sample]
                y_bt = y_arr[idx_bt]
                X_bt_orig = [X_test[i] for i in idx_bt]
                orig_bt = {name: fn_(X_bt_orig).argmax(1) for name, fn_ in models.items()}
                X_bt = [bt(t) for t in tqdm(X_bt_orig, desc="BT")]
                pert = {name: fn_(X_bt).argmax(1) for name, fn_ in models.items()}
                for name in models:
                    rows.append(
                        _row(name, pert_name, len(y_bt), y_bt, orig_bt[name], pert[name])
                    )
            else:
                X_pert = fn(X_orig)
                pert = {name: fn_(X_pert).argmax(1) for name, fn_ in models.items()}
                for name in models:
                    rows.append(_row(name, pert_name, n, y, orig_preds[name], pert[name]))
    finally:
        if bt is not None:
            bt.unload()

    df = pd.DataFrame(rows)
    df["ΔF1"] = df["F1_orig"] - df["F1_pert"]
    df = df.round(4)
    print("\n=== Robustez adversarial ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


def _row(model: str, pert: str, n: int, y, orig_p, pert_p) -> dict:
    return {
        "Model": model,
        "Perturbation": pert,
        "N": n,
        "F1_orig": f1_score(y, orig_p, average="macro", zero_division=0),
        "F1_pert": f1_score(y, pert_p, average="macro", zero_division=0),
        "flip_rate": float((orig_p != pert_p).mean()),
    }


__all__ = [
    "ProbaPredictor",
    "run_adversarial_eval",
]
