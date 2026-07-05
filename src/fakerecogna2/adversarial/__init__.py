"""Robustez adversarial: perturbações de texto + back-translation."""

from . import back_translation, perturbations, runner
from .back_translation import BackTranslator
from .perturbations import adv_random_typos, adv_word_deletion, adv_word_swap
from .runner import ProbaPredictor, run_adversarial_eval

__all__ = [
    "perturbations",
    "back_translation",
    "runner",
    "adv_random_typos",
    "adv_word_deletion",
    "adv_word_swap",
    "BackTranslator",
    "ProbaPredictor",
    "run_adversarial_eval",
]
