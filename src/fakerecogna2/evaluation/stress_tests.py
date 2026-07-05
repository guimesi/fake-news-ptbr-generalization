"""Stress tests: split por fonte, NER masking, split temporal.

A maior parte da orquestração (treinar CNN+LSTM em splits alternativos) é
delegada pra `models.train_ensemble_on_variant`. Aqui ficam só as utilidades
específicas (mascaramento NER, avaliação de modelo treinado em loader alternativo).
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..config import BATCH_SIZE
from ..preprocessing.text_cleaning import get_spacy_pt, preprocess_base
from ..utils.io_utils import save_table




# -- API limpa ----------------------------------------------------------------
def mask_named_entities(
    text: str,
    nlp=None,
    mask: str = "[ENT]",
    labels: tuple[str, ...] = ("PER", "ORG", "LOC"),
) -> str:
    """Substitui entidades nomeadas pelo placeholder `mask`."""
    nlp = nlp or get_spacy_pt()
    doc = nlp(str(text)[:5000])
    out, last = [], 0
    for ent in doc.ents:
        if ent.label_ in labels:
            out.append(text[last : ent.start_char])
            out.append(mask)
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


def build_masked_test_loader(
    X_test: list[str],
    y_test: np.ndarray,
    extractor,
    batch_size: int = BATCH_SIZE,
    apply_preprocess: bool = True,
) -> DataLoader:
    """Aplica NER masking → preprocess_base → extract_token_embs → DataLoader."""
    if apply_preprocess:
        masked = [preprocess_base(mask_named_entities(t)) for t in tqdm(X_test, desc="NER-mask")]
    else:
        masked = [mask_named_entities(t) for t in tqdm(X_test, desc="NER-mask")]
    emb = extractor.extract_token_embs(masked)
    return DataLoader(
        TensorDataset(emb, torch.tensor(np.asarray(y_test), dtype=torch.long)),
        batch_size=batch_size,
    )


def evaluate_models_on_loader(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Avalia cada modelo no loader; retorna dict nome→predições."""
    preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        ps = []
        with torch.no_grad():
            for bx, _ in loader:
                p = torch.softmax(model(bx.to(device)), 1).cpu()
                ps.extend(p.argmax(1).numpy())
        preds[name] = np.array(ps)
    return preds


def ner_ablation_table(
    y_test: np.ndarray,
    preds_orig: dict[str, np.ndarray],
    preds_masked: dict[str, np.ndarray],
    save_as: str | None = "14_ner_masking",
    print_table: bool = True,
) -> pd.DataFrame:
    """Tabela F1 orig vs F1 mascarado por modelo."""
    rows = []
    for name in preds_orig.keys() & preds_masked.keys():
        f1_o = f1_score(y_test, preds_orig[name], average="macro", zero_division=0)
        f1_m = f1_score(y_test, preds_masked[name], average="macro", zero_division=0)
        rows.append(
            {"Model": name, "F1 (orig)": f1_o, "F1 (NER-masked)": f1_m, "Δ F1": f1_o - f1_m}
        )
    df = pd.DataFrame(rows).round(4)
    if print_table:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


def compare_split_results(
    baseline_acc: float,
    baseline_f1: float,
    variant_acc: float,
    variant_f1: float,
    baseline_label: str = "Random",
    variant_label: str = "Source-split",
    save_as: str | None = None,
) -> pd.DataFrame:
    """Tabelinha Random vs Variant (Acc, F1): usada por split por fonte e split temporal."""
    df = pd.DataFrame(
        [
            {"Split": baseline_label, "Accuracy": baseline_acc, "F1": baseline_f1},
            {"Split": variant_label, "Accuracy": variant_acc, "F1": variant_f1},
        ]
    ).round(4)
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


__all__ = [
    "mask_named_entities",
    "build_masked_test_loader",
    "evaluate_models_on_loader",
    "ner_ablation_table",
    "compare_split_results",
]
