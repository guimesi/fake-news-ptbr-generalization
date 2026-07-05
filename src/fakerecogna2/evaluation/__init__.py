"""Métricas, bootstrap, calibração, CV, stress tests, análise de erros, cross-dataset, ablações."""

from . import (
    ablations,
    bootstrap,
    calibration,
    confusion_matrices,
    cross_dataset,
    cross_validation,
    error_analysis,
    metrics,
    plots,
    stress_tests,
    xai_error_integration,
)
from .ablations import (
    learning_curve_ablation,
    performance_by_length_ensemble,
    preprocessing_ablation,
    seqlen_ablation,
    short_quartile_class_distribution,
)
from .bootstrap import bootstrap_ci, bootstrap_table, f1_macro
from .calibration import calibration_table, expected_calibration_error, plot_reliability
from .confusion_matrices import (
    per_class_metrics,
    plot_confusion_matrix,
    save_all_cms,
    save_cm,
)
from .cross_dataset import (
    encode_external_labels,
    error_diagnosis_cross_dataset,
    evaluate_on_external_corpus,
    save_polarity_audit_log,
    train_on_fakebr_eval_main,
)
from .cross_validation import cross_validate_ensemble
from .error_analysis import (
    error_distribution_full,
    error_summary,
    performance_by_length,
)
from .xai_error_integration import (
    aggregate_lime_tokens_by_dim_cell,
    build_cell_per_example,
    consolidate_xai_errors,
    cross_cell_by_dimension,
)
from .metrics import standard_report
from .plots import plot_final_comparison
from .stress_tests import (
    build_masked_test_loader,
    compare_split_results,
    evaluate_models_on_loader,
    mask_named_entities,
    ner_ablation_table,
)

__all__ = [
    "metrics",
    "plots",
    "bootstrap",
    "calibration",
    "confusion_matrices",
    "xai_error_integration",
    "cross_validation",
    "stress_tests",
    "error_analysis",
    "cross_dataset",
    "ablations",
    "standard_report",
    "bootstrap_ci",
    "f1_macro",
    "bootstrap_table",
    "expected_calibration_error",
    "plot_reliability",
    "calibration_table",
    "per_class_metrics",
    "plot_confusion_matrix",
    "save_cm",
    "save_all_cms",
    "cross_validate_ensemble",
    "mask_named_entities",
    "build_masked_test_loader",
    "evaluate_models_on_loader",
    "ner_ablation_table",
    "compare_split_results",
    "error_summary",
    "performance_by_length",
    "error_distribution_full",
    "build_cell_per_example",
    "cross_cell_by_dimension",
    "aggregate_lime_tokens_by_dim_cell",
    "consolidate_xai_errors",
    "encode_external_labels",
    "evaluate_on_external_corpus",
    "error_diagnosis_cross_dataset",
    "save_polarity_audit_log",
    "train_on_fakebr_eval_main",
    "plot_final_comparison",
    "preprocessing_ablation",
    "seqlen_ablation",
    "learning_curve_ablation",
    "performance_by_length_ensemble",
    "short_quartile_class_distribution",
]
