"""Configuração centralizada do projeto FakeRecogna 2.0.

Fonte de verdade dos parâmetros: o arquivo `configs/config.yaml`, carregado no
import. As constantes abaixo (SEED, MAX_LEN, BATCH_SIZE, paths, lista de PLMs,
etc.) são lidas do YAML; o valor `default=` em cada `_g(...)` é apenas um
fallback usado se o YAML estiver ausente ou se a chave não existir. Os fallbacks
são mantidos idênticos aos valores do YAML para que o protocolo experimental se
preserve mesmo sem o arquivo.

Edite o YAML, não este arquivo, para mudar o protocolo experimental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML; opcional
except ImportError:  # pragma: no cover
    yaml = None

# --- Raiz do projeto ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CFG = _load_yaml(CONFIG_PATH)


def _g(*keys: str, default=None):
    cur: Any = _CFG
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --- Determinismo ------------------------------------------------------------
SEED: int = int(_g("experiment", "seed", default=42))
SEEDS_MULTI: list[int] = list(_g("experiment", "seeds_multi", default=[42, 7, 2024]))

# --- Dataset -----------------------------------------------------------------
DATASET_HF_ID_TEMPLATE: str = _g(
    "data", "hf_id_template", default="recogna-nlp/fakerecogna2-{variant}"
)
TEXT_COLUMNS_ABSTRATIVA: list[str] = list(
    _g("data", "text_columns_abstrativa", default=["news_text_full", "text", "noticia"])
)
TEXT_COLUMNS_EXTRATIVA: list[str] = list(
    _g("data", "text_columns_extrativa", default=["news_text_short", "summary", "resumo"])
)
LABEL_COLUMN: str = _g("data", "label_column", default="label")

# --- Splits ------------------------------------------------------------------
TEST_SIZE: float = float(_g("splits", "test_size", default=0.20))
VAL_SIZE: float = float(_g("splits", "val_size", default=0.10))

# --- Modelos / BERTimbau -----------------------------------------------------
BERTIMBAU_MODEL: str = _g("models", "bertimbau", default="neuralmind/bert-base-portuguese-cased")
MAX_LEN: int = int(_g("models", "max_len", default=200))
BATCH_SIZE: int = int(_g("models", "batch_size", default=32))
NUM_EPOCHS: int = int(_g("models", "num_epochs", default=10))
LEARNING_RATE: float = float(_g("models", "learning_rate", default=2e-5))
PATIENCE: int = int(_g("models", "patience", default=3))

# --- PLMs adicionais ---------------------------------------------
PLM_CANDIDATES: list[str] = list(
    _g(
        "models",
        "plm_candidates",
        default=[
            "neuralmind/bert-large-portuguese-cased",
            "xlm-roberta-base",
            "microsoft/mdeberta-v3-base",
        ],
    )
)

# --- Avaliação estatística --------------------------------------------------
BOOTSTRAP_ITERS: int = int(_g("evaluation", "bootstrap_iters", default=10000))
CALIBRATION_BINS: int = int(_g("evaluation", "calibration_bins", default=15))
RELIABILITY_BINS: int = int(_g("evaluation", "reliability_bins", default=10))
CV_FOLDS: int = int(_g("evaluation", "cv_folds", default=5))

# --- Robustez adversarial ---------------------------------------------------
BT_PT_TO_EN_MODEL: str = _g(
    "adversarial", "pt_to_en_model", default="Helsinki-NLP/opus-mt-roa-en"
)
BT_EN_TO_PT_MODEL: str = _g(
    "adversarial", "en_to_pt_model", default="Helsinki-NLP/opus-mt-en-roa"
)
ADV_TYPO_RATE: float = float(_g("adversarial", "typo_rate", default=0.05))
ADV_DELETION_RATE: float = float(_g("adversarial", "deletion_rate", default=0.10))
ADV_SWAP_RATE: float = float(_g("adversarial", "swap_rate", default=0.05))
ADV_PERTURB_SAMPLE_SIZE: int = int(_g("adversarial", "perturb_sample_size", default=500))
ADV_BT_SAMPLE_SIZE: int = int(_g("adversarial", "bt_sample_size", default=300))

# --- Diretórios de artefatos ------------------------------------------------
ARTIFACTS_DIR: Path = Path(_g("paths", "artifacts_dir", default=str(PROJECT_ROOT / "outputs")))
DATA_RAW_DIR: Path = Path(_g("paths", "data_raw", default=str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED_DIR: Path = Path(_g("paths", "data_processed", default=str(PROJECT_ROOT / "data" / "processed")))
DATA_INTERIM_DIR: Path = Path(_g("paths", "data_interim", default=str(PROJECT_ROOT / "data" / "interim")))
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
TABLES_DIR: Path = ARTIFACTS_DIR / "metrics"
PLOTS_DIR: Path = ARTIFACTS_DIR / "figures"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
LIME_DIR: Path = ARTIFACTS_DIR / "explanations"

# --- Fake.br-Corpus (cross-dataset) -----------------------------------------
FAKEBR_LOCAL_PATH: Path = Path(
    _g(
        "data",
        "fakebr_local_path",
        default=str(PROJECT_ROOT / "data" / "external" / "Fake.br-Corpus-master"),
    )
)


@dataclass
class ExperimentConfig:
    """Snapshot completo dos parâmetros do experimento (útil para logging)."""

    seed: int = SEED
    seeds_multi: list[int] = field(default_factory=lambda: list(SEEDS_MULTI))
    dataset_template: str = DATASET_HF_ID_TEMPLATE
    bertimbau: str = BERTIMBAU_MODEL
    max_len: int = MAX_LEN
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    patience: int = PATIENCE
    test_size: float = TEST_SIZE
    val_size: float = VAL_SIZE
    bootstrap_iters: int = BOOTSTRAP_ITERS
    cv_folds: int = CV_FOLDS
    artifacts_dir: str = str(ARTIFACTS_DIR)


def ensure_dirs() -> None:
    """Cria toda a árvore de outputs se ainda não existir."""
    for d in [
        ARTIFACTS_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_INTERIM_DIR,
        MODELS_DIR,
        TABLES_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        LIME_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "SEED",
    "SEEDS_MULTI",
    "DATASET_HF_ID_TEMPLATE",
    "TEXT_COLUMNS_ABSTRATIVA",
    "TEXT_COLUMNS_EXTRATIVA",
    "LABEL_COLUMN",
    "TEST_SIZE",
    "VAL_SIZE",
    "BERTIMBAU_MODEL",
    "MAX_LEN",
    "BATCH_SIZE",
    "NUM_EPOCHS",
    "LEARNING_RATE",
    "PATIENCE",
    "PLM_CANDIDATES",
    "BOOTSTRAP_ITERS",
    "CALIBRATION_BINS",
    "RELIABILITY_BINS",
    "CV_FOLDS",
    "BT_PT_TO_EN_MODEL",
    "BT_EN_TO_PT_MODEL",
    "ADV_TYPO_RATE",
    "ADV_DELETION_RATE",
    "ADV_SWAP_RATE",
    "ADV_PERTURB_SAMPLE_SIZE",
    "ADV_BT_SAMPLE_SIZE",
    "ARTIFACTS_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_INTERIM_DIR",
    "MODELS_DIR",
    "TABLES_DIR",
    "PLOTS_DIR",
    "LOGS_DIR",
    "LIME_DIR",
    "FAKEBR_LOCAL_PATH",
    "ExperimentConfig",
    "ensure_dirs",
]
