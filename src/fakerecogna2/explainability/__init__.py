"""XAI: LIME, Integrated Gradients, Attention Rollout, comparação, estabilidade."""

from . import attention_rollout, comparison, integrated_gradients, lime_explainer, stability
from .attention_rollout import AttentionRollout
from .comparison import compare_xai_for_examples
from .integrated_gradients import compute_integrated_gradients
from .lime_explainer import (
    ProbaPredictor,
    aggregate_lime_tokens_by_tag,
    run_lime_explanations,
    select_lime_targets,
    select_lime_targets_4cells,
)
from .stability import lime_stability

__all__ = [
    "lime_explainer",
    "integrated_gradients",
    "attention_rollout",
    "comparison",
    "stability",
    "ProbaPredictor",
    "select_lime_targets",
    "select_lime_targets_4cells",
    "aggregate_lime_tokens_by_tag",
    "run_lime_explanations",
    "compute_integrated_gradients",
    "AttentionRollout",
    "compare_xai_for_examples",
    "lime_stability",
]
