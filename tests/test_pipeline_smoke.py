"""Smoke test do pipeline.

Valida que:
- todos os subpacotes esperados importam sem erro,
- `fakerecogna2.config` carrega com os defaults corretos,
- `ExperimentContext` instancia OK.
"""

from __future__ import annotations

import importlib

import pytest


EXPECTED_SUBPACKAGES = [
    "fakerecogna2.data",
    "fakerecogna2.preprocessing",
    "fakerecogna2.features",
    "fakerecogna2.models",
    "fakerecogna2.evaluation",
    "fakerecogna2.statistics",
    "fakerecogna2.explainability",
    "fakerecogna2.adversarial",
    "fakerecogna2.deployment",
    "fakerecogna2.reports",
    "fakerecogna2.utils",
]


@pytest.mark.parametrize("name", EXPECTED_SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)


def test_config_loads():
    from fakerecogna2 import config

    assert config.SEED == 42
    assert config.MAX_LEN > 0
    assert config.BERTIMBAU_MODEL.startswith("neuralmind/")


def test_experiment_context_instantiates():
    from fakerecogna2 import ExperimentContext

    ctx = ExperimentContext()
    assert ctx.seed == 42
    assert ctx.device in ("cuda", "cpu")
    assert ctx.df is None
    assert isinstance(ctx.extras, dict)
