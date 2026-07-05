"""Relatório consolidado em Markdown e JSON."""

from . import consolidated
from .consolidated import (
    build_consolidated_report,
    build_final_table,
    build_manifest,
    build_results_json,
    print_classification_report,
)

__all__ = [
    "consolidated",
    "build_final_table",
    "print_classification_report",
    "build_manifest",
    "build_consolidated_report",
    "build_results_json",
]
