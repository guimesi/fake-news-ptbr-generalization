"""Configuração de logging consistente com a célula 1.3 do notebook.

A célula original cria um logger 'FN' que escreve em `artifacts/logs/exp_*.log`
e também em stdout. Aqui replicamos o comportamento, redirecionando para
`outputs/logs/`.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from ..config import LOGS_DIR

_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S"
)
_LOGGER_NAME = "FN"
_INITIALIZED = False


def setup_logging(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Cria/atualiza o logger 'FN' replicando a célula 1.3.

    Retorna o logger já configurado com handler de arquivo (em outputs/logs/) e
    handler de stdout.
    """
    global _INITIALIZED
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"exp_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(_FORMATTER)
        logger.addHandler(h)
    logger.propagate = False
    _INITIALIZED = True
    logger.info(f"Logging inicializado. Arquivo: {log_path.name}")
    return logger


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Retorna o logger global, inicializando-o sob demanda."""
    if not _INITIALIZED:
        return setup_logging(name)
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
