"""Loop de treinamento + avaliação + multi-seed (cells 32 e 33)."""

from __future__ import annotations

import time
from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import SEED, SEEDS_MULTI
from ..utils.io_utils import RESULTS, RESULTS_MULTISEED
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def train_model(
    model: nn.Module,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    device: str = "cpu",
    epochs: int = 30,
    lr: float = 1e-3,
    es_patience: int = 5,
    sched_patience: int = 2,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    model_name: str = "model",
    seed: int = SEED,
) -> tuple[nn.Module, dict]:
    """Treina com Adam + ReduceLROnPlateau + early-stop.

    Restaura os pesos da melhor época (menor val_loss) antes de retornar.
    """
    set_seed(seed)
    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=sched_patience, factor=0.5
    )
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_vl, best_st, pat = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tl, c, n = 0.0, 0, 0
        for bx, by in tqdm(tr_loader, desc=f"{model_name} ep{ep+1}", leave=False):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            tl += loss.item() * bx.size(0)
            c += (out.argmax(1) == by).sum().item()
            n += bx.size(0)
        tl /= n
        ta = c / n

        model.eval()
        vl, c, n = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in vl_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                vl += crit(out, by).item() * bx.size(0)
                c += (out.argmax(1) == by).sum().item()
                n += bx.size(0)
        vl /= n
        va = c / n
        sched.step(vl)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        log.info(
            f"{model_name} ep{ep+1}: trL={tl:.4f} trA={ta:.4f} vlL={vl:.4f} vlA={va:.4f}"
        )

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= es_patience:
                log.info(f"{model_name}: early stop ep {ep+1}")
                break

    if best_st:
        model.load_state_dict(best_st)
    return model, hist


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    name: str = "model",
) -> tuple[dict, np.ndarray, np.ndarray, torch.Tensor]:
    """Avalia e retorna (métricas, preds, labels, probs)."""
    model.eval()
    ps, ls, probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bx, by in loader:
            out = model(bx.to(device))
            p = torch.softmax(out, 1).cpu()
            probs.append(p)
            ps.extend(p.argmax(1).numpy())
            ls.extend(by.numpy())
    inf = (time.time() - t0) / max(1, len(ls))
    probs = torch.cat(probs, 0)
    m = {
        "Accuracy": accuracy_score(ls, ps),
        "Precision": precision_score(ls, ps, average="macro", zero_division=0),
        "Recall": recall_score(ls, ps, average="macro", zero_division=0),
        "F1": f1_score(ls, ps, average="macro", zero_division=0),
        "Inference (ms)": inf * 1000,
    }
    log.info(f'{name}: Acc={m["Accuracy"]:.4f} F1={m["F1"]:.4f}')
    return m, np.array(ps), np.array(ls), probs


def train_multiseed(
    model_cls: Type[nn.Module],
    loader_tr: DataLoader,
    loader_vl: DataLoader,
    loader_te: DataLoader,
    kwargs: dict[str, Any],
    seeds: list[int] = SEEDS_MULTI,
    epochs: int = 30,
    device: str = "cpu",
    model_name: str = "Model",
) -> tuple[list[dict], list[torch.Tensor], list[dict]]:
    """Treina `model_cls` com cada seed e devolve metrics/probs/hist por seed."""
    metrics_list, probs_list, hist_list = [], [], []
    for s in seeds:
        log.info(f"--- {model_name} seed={s} ---")
        m = model_cls(**kwargs)
        m, h = train_model(
            m, loader_tr, loader_vl, device=device, epochs=epochs,
            model_name=f"{model_name}_s{s}", seed=s,
        )
        met, _, _, probs = evaluate_model(m, loader_te, device=device, name=f"{model_name}_s{s}")
        metrics_list.append(met)
        probs_list.append(probs)
        hist_list.append(h)
    return metrics_list, probs_list, hist_list


def aggregate_multiseed_results(metrics_list: list[dict], name: str) -> None:
    """Agrega métricas de múltiplas seeds → RESULTS_MULTISEED (mean, std) + RESULTS[seed0]."""
    import pandas as pd

    df = pd.DataFrame(metrics_list)
    out = {
        c: (float(df[c].mean()), float(df[c].std()))
        for c in ["Accuracy", "Precision", "Recall", "F1", "Inference (ms)"]
        if c in df.columns
    }
    RESULTS_MULTISEED[name] = out
    RESULTS[name] = metrics_list[0]
    log.info(
        f'{name}: F1 = {out["F1"][0]:.4f} ± {out["F1"][1]:.4f} '
        f"(n={len(metrics_list)} seeds)"
    )


def avg_probs(probs_list: list[torch.Tensor]) -> torch.Tensor:
    """Média das probabilidades em uma lista de tensores [N, num_classes]."""
    return torch.stack(probs_list, 0).mean(0)


__all__ = [
    "train_model",
    "evaluate_model",
    "train_multiseed",
    "aggregate_multiseed_results",
    "avg_probs",
]
