"""Baselines, redes neurais, ensembles, BERTimbau fine-tuned e PLMs maiores."""

from . import baselines, bertimbau_finetune, ensembles, neural_models, plm_finetune, training
from .baselines import (
    make_baseline_configs,
    train_baselines,
    train_baselines_multiseed,
)
from .bertimbau_finetune import (
    BERTClassifier,
    make_bert_loader,
    train_bertimbau_finetune,
)
from .ensembles import (
    get_val_probs,
    grid_search_2model,
    grid_search_3model,
    register_ensemble,
    train_ensemble_on_variant,
)
from .neural_models import TextCNN, TextConvLSTM, TextLSTM
from .plm_finetune import evaluate_plm_candidates, fine_tune_plm
from .training import (
    aggregate_multiseed_results,
    avg_probs,
    evaluate_model,
    train_model,
    train_multiseed,
)

__all__ = [
    "baselines",
    "neural_models",
    "training",
    "ensembles",
    "bertimbau_finetune",
    "plm_finetune",
    "make_baseline_configs",
    "train_baselines",
    "train_baselines_multiseed",
    "TextCNN",
    "TextLSTM",
    "TextConvLSTM",
    "train_model",
    "evaluate_model",
    "train_multiseed",
    "aggregate_multiseed_results",
    "avg_probs",
    "get_val_probs",
    "grid_search_2model",
    "grid_search_3model",
    "register_ensemble",
    "train_ensemble_on_variant",
    "BERTClassifier",
    "make_bert_loader",
    "train_bertimbau_finetune",
    "fine_tune_plm",
    "evaluate_plm_candidates",
]
