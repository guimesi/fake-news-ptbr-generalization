"""Estatísticas e testes: log-odds Dirichlet, Chi², McNemar pairwise Holm."""

from . import log_odds, significance_tests
from .log_odds import (
    build_ngram_counts,
    chi2_top,
    lex_analysis,
    log_odds_dirichlet,
    show_examples_for_term,
)
from .significance_tests import MCNEMAR_FAMILY, mcnemar_pair, mcnemar_pairwise_holm

__all__ = [
    "log_odds",
    "significance_tests",
    "log_odds_dirichlet",
    "build_ngram_counts",
    "lex_analysis",
    "chi2_top",
    "show_examples_for_term",
    "MCNEMAR_FAMILY",
    "mcnemar_pair",
    "mcnemar_pairwise_holm",
]
