"""Métricas práticas para deployment: latência, tamanho em disco, fronteira de Pareto."""

from . import benchmarks
from .benchmarks import (
    audit_model_parameters,
    benchmark_latency,
    benchmark_models,
    count_parameters,
    measure_disk_sizes,
    measure_pytorch_model_size,
    measure_sklearn_object_size,
    measure_vram_peak,
    model_disk_size_mb,
    plot_pareto_f1_latency,
)

__all__ = [
    "benchmarks",
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
