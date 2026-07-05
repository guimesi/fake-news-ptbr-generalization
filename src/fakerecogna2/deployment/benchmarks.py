"""Métricas de deployment: latência (P50/P95/P99), VRAM, tamanho em disco, Pareto."""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..config import ARTIFACTS_DIR, SEED
from ..utils.io_utils import save_plot, save_table
from ..utils.logging_utils import get_logger

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def benchmark_latency(
    predict_fn: Callable[[list[str]], np.ndarray],
    sample_texts: list[str],
    n_warmup: int = 10,
    n_iter: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, float]:
    """Mede latência com warmup + cuda.synchronize. P50/P95/P99 em ms/amostra."""
    rng = random.Random(seed)
    is_cuda = _is_cuda(device)

    for _ in range(n_warmup):
        batch = rng.sample(sample_texts, min(batch_size, len(sample_texts)))
        predict_fn(batch)
        if is_cuda:
            _cuda_sync()

    times: list[float] = []
    for _ in range(n_iter):
        batch = rng.sample(sample_texts, min(batch_size, len(sample_texts)))
        if is_cuda:
            _cuda_sync()
        t0 = time.time()
        predict_fn(batch)
        if is_cuda:
            _cuda_sync()
        times.append((time.time() - t0) * 1000 / batch_size)

    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "throughput_sps": 1000.0 / float(np.mean(times)),
        "n_iter": n_iter,
        "batch_size": batch_size,
    }


def measure_vram_peak(
    predict_fn: Callable[[list[str]], np.ndarray],
    sample_texts: list[str],
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """VRAM pico (MB) durante uma inferência. Retorna 0 em CPU."""
    if not _is_cuda(device):
        return 0.0
    import torch

    torch.cuda.reset_peak_memory_stats()
    predict_fn(sample_texts[:batch_size])
    return torch.cuda.max_memory_allocated() / 1e6


def model_disk_size_mb(path: str | Path) -> float:
    """Tamanho do arquivo em MB. Retorna 0 se não existir."""
    p = Path(path)
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def count_parameters(model) -> int:
    """Conta parâmetros treináveis de um modelo torch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_pytorch_model_size(model) -> float:
    """Salva state_dict num temp file, mede tamanho, deleta. Retorna MB."""
    import torch

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    try:
        torch.save(model.state_dict(), tmp.name)
        tmp.close()
        return os.path.getsize(tmp.name) / 1e6
    finally:
        os.unlink(tmp.name)


def measure_sklearn_object_size(obj) -> float:
    """Salva objeto sklearn (vectorizer, classifier) via joblib, mede MB."""
    import joblib

    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    try:
        joblib.dump(obj, tmp.name)
        tmp.close()
        return os.path.getsize(tmp.name) / 1e6
    finally:
        os.unlink(tmp.name)


def benchmark_models(
    benchmark_pool: list[str],
    predict_fns: dict[str, Callable[[list[str]], np.ndarray]],
    extras: dict[str, dict] | None = None,
    n_warmup: int = 20,
    n_iter: int = 200,
    device: str = "cpu",
    save_as: str | None = "22_deployment_metrics",
) -> pd.DataFrame:
    """Roda latência + VRAM + extras (params, disk, F1) pra cada modelo.

    Args:
        benchmark_pool: lista de textos do test (use ~200).
        predict_fns: dict nome→função predict.
        extras: dict nome→dict de campos adicionais (ex.: F1, Params, Disk).
        n_warmup/n_iter: parâmetros do benchmark_latency.
        device: 'cpu' ou 'cuda'.

    Returns:
        DataFrame com colunas Model, F1, p50/p95/p99_ms, throughput, VRAM, Params, Disk.
    """
    extras = extras or {}
    rows: list[dict] = []
    for name, fn in predict_fns.items():
        bench = benchmark_latency(
            fn, benchmark_pool, n_warmup=n_warmup, n_iter=n_iter, batch_size=1,
            device=device,
        )
        vram = measure_vram_peak(fn, benchmark_pool, batch_size=32, device=device)
        rows.append({"Model": name, **bench, "VRAM peak (MB)": vram, **extras.get(name, {})})

    df = pd.DataFrame(rows)
    cols = [
        "Model", "F1", "p50_ms", "p95_ms", "p99_ms", "throughput_sps",
        "VRAM peak (MB)", "Params (M)", "Disk (MB)",
    ]
    df = df[[c for c in cols if c in df.columns]].round(3)
    print("\n=== Métricas de deployment ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


def plot_pareto_f1_latency(
    df_bench: pd.DataFrame,
    f1_col: str = "F1",
    latency_col: str = "p50_ms",
    save_as: str | None = "22_pareto_f1_vs_latency",
) -> None:
    """Scatter F1 × latência (eixo x em log). Anota cada modelo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df_bench.iterrows():
        if pd.isna(row.get(f1_col)):
            continue
        ax.scatter(row[latency_col], row[f1_col] * 100, s=200, alpha=0.7)
        ax.annotate(
            row["Model"],
            (row[latency_col], row[f1_col] * 100),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latência P50 (ms/amostra), escala log")
    ax.set_ylabel("F1 macro (%)")
    ax.set_title(
        "Trade-off F1 × Latência: escolha do modelo em produção", fontweight="bold"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_as is not None:
        save_plot(fig, save_as)


def measure_disk_sizes(
    models: dict[str, object],
    save_as: str | None = "J_disk_sizes_corrected",
) -> pd.DataFrame:
    """Mede tamanho em disco de cada modelo.

    Args:
        models: dict nome→objeto. Detecta automaticamente se é torch.nn.Module
            (state_dict) ou sklearn (joblib). Listas/tuples de modelos somam tamanhos.

    Returns:
        DataFrame Model, Disk (MB).
    """
    disk_sizes: dict[str, float] = {}
    for name, obj in models.items():
        if isinstance(obj, (list, tuple)):
            disk_sizes[name] = sum(_smart_model_size(m) for m in obj)
        else:
            disk_sizes[name] = _smart_model_size(obj)

    df = pd.DataFrame([{"Model": k, "Disk (MB)": round(v, 2)} for k, v in disk_sizes.items()])
    print("\n=== Tamanho real em disco dos modelos ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        out_path = ARTIFACTS_DIR / "metrics" / f"{save_as}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(df.to_csv(index=False), encoding="utf-8")
    return df


def audit_model_parameters(
    models: dict[str, object],
    save_as: str | None = "J_parameters_audit",
) -> pd.DataFrame:
    """Conta parametros (totais + treinaveis) por modelo.

    Args:
        models: dict nome -> objeto. Aceita listas/tuples (soma os subitens).
    """
    import torch

    rows = []
    for name, obj in models.items():
        if isinstance(obj, (list, tuple)):
            total = sum(
                sum(p.numel() for p in m.parameters())
                for m in obj if isinstance(m, torch.nn.Module)
            )
            train = sum(
                sum(p.numel() for p in m.parameters() if p.requires_grad)
                for m in obj if isinstance(m, torch.nn.Module)
            )
        elif isinstance(obj, torch.nn.Module):
            total = sum(p.numel() for p in obj.parameters())
            train = sum(p.numel() for p in obj.parameters() if p.requires_grad)
        else:
            # sklearn: tenta inferir via coef_/intercept_ shape
            n_coef = 0
            for attr in ("coef_", "intercept_", "support_vectors_"):
                if hasattr(obj, attr):
                    arr = getattr(obj, attr)
                    if arr is not None and hasattr(arr, "size"):
                        n_coef += int(arr.size)
            total = train = n_coef
        rows.append({
            "Model": name,
            "Params total": int(total),
            "Params trainable": int(train),
            "Params (M)": round(total / 1e6, 3),
        })
    df = pd.DataFrame(rows)
    print("\n=== Auditoria de parametros ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)
    return df


def _smart_model_size(obj) -> float:
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return measure_pytorch_model_size(obj)
    except ImportError:
        pass
    return measure_sklearn_object_size(obj)


__all__ = [
    "benchmark_latency",
    "measure_vram_peak",
    "model_disk_size_mb",
    "count_parameters",
    "measure_pytorch_model_size",
    "measure_sklearn_object_size",
    "benchmark_models",
    "plot_pareto_f1_latency",
    "measure_disk_sizes",
    "audit_model_parameters",
]
