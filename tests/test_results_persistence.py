"""Testa que RESULTS é um dict auto-persistente entre processos."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _preserve_multiseed_cache():
    """clear_results() também esvazia results_multiseed.json (also_multiseed=True).

    Os testes deste módulo só preservavam results.json, então cada execução do
    pytest zerava o cache multiseed real. Este fixture faz backup/restauração
    do arquivo em torno de todos os testes do módulo.
    """
    ms_file = ROOT / "outputs" / ".cache" / "results_multiseed.json"
    backup = ms_file.read_bytes() if ms_file.exists() else None
    yield
    if backup is not None:
        ms_file.write_bytes(backup)
    elif ms_file.exists():
        ms_file.unlink()


def _run_in_subprocess(code: str, env: dict | None = None) -> str:
    """Roda um snippet Python em subprocesso, retorna stdout."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_results_is_dict_subclass():
    """RESULTS continua expondo interface de dict (subclasse direta)."""
    from fakerecogna2.utils import RESULTS

    assert isinstance(RESULTS, dict)


def test_results_persists_across_processes(tmp_path, monkeypatch):
    """Escrever em RESULTS num processo deve aparecer no próximo."""
    cache_dir = ROOT / "outputs" / ".cache"
    cache_file = cache_dir / "results.json"

    # Limpa antes de testar (preserva backup se existir)
    backup = None
    if cache_file.exists():
        backup = cache_file.read_bytes()
        cache_file.unlink()

    try:
        # Processo 1: escreve
        _run_in_subprocess(
            "from fakerecogna2.utils import RESULTS, clear_results\n"
            "clear_results()\n"
            "RESULTS['test_baseline'] = {'F1': 0.85, 'Accuracy': 0.88}\n"
            "RESULTS['test_deep'] = {'F1': 0.92, 'Accuracy': 0.93}\n"
            "print('ok-write')"
        )
        assert cache_file.exists(), "Cache file não foi criado"
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "test_baseline" in data
        assert data["test_baseline"]["F1"] == 0.85

        # Processo 2: lê
        out = _run_in_subprocess(
            "from fakerecogna2.utils import RESULTS\n"
            "print(list(RESULTS.keys()))\n"
            "print(RESULTS.get('test_baseline', {}).get('F1'))"
        )
        assert "test_baseline" in out
        assert "test_deep" in out
        assert "0.85" in out
    finally:
        # Limpa o teste e restaura backup
        if cache_file.exists():
            cache_file.unlink()
        if backup is not None:
            cache_file.write_bytes(backup)


def test_results_fresh_env_skips_load(tmp_path):
    """FAKERECOGNA_RESULTS_FRESH=1 deve ignorar cache existente."""
    cache_dir = ROOT / "outputs" / ".cache"
    cache_file = cache_dir / "results.json"

    backup = cache_file.read_bytes() if cache_file.exists() else None

    try:
        # Semeia o cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"old_model": {"F1": 0.5}}), encoding="utf-8"
        )

        # Processo com FRESH=1 não deve ver old_model
        out = _run_in_subprocess(
            "from fakerecogna2.utils import RESULTS\n"
            "print('old_model' in RESULTS)\n"
            "print(len(RESULTS))",
            env={"FAKERECOGNA_RESULTS_FRESH": "1"},
        )
        assert "False" in out
        assert "\n0\n" in out or out.strip().endswith("0")
    finally:
        if cache_file.exists():
            cache_file.unlink()
        if backup is not None:
            cache_file.write_bytes(backup)


def test_clear_results_removes_cache():
    """clear_results() esvazia memória + disco."""
    from fakerecogna2.utils import RESULTS, clear_results

    cache_file = ROOT / "outputs" / ".cache" / "results.json"
    backup = cache_file.read_bytes() if cache_file.exists() else None

    try:
        clear_results()
        RESULTS["sentinel"] = {"F1": 0.7}
        assert "sentinel" in RESULTS
        assert cache_file.exists()

        clear_results()
        assert len(RESULTS) == 0
        # Após clear, o JSON em disco deve estar vazio
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data == {}
    finally:
        if backup is not None:
            cache_file.write_bytes(backup)


def test_results_handles_numpy_types():
    """np.float32 e np.int64 devem ser serializáveis sem explodir."""
    import numpy as np

    from fakerecogna2.utils import RESULTS, clear_results

    cache_file = ROOT / "outputs" / ".cache" / "results.json"
    backup = cache_file.read_bytes() if cache_file.exists() else None

    try:
        clear_results()
        RESULTS["np_model"] = {
            "F1": np.float32(0.9234),
            "N": np.int64(1000),
            "preds": np.array([0, 1, 0, 1]),
        }
        # Lê de volta — não deve falhar
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "np_model" in data
        assert abs(data["np_model"]["F1"] - 0.9234) < 1e-4
        assert data["np_model"]["N"] == 1000
        assert data["np_model"]["preds"] == [0, 1, 0, 1]
    finally:
        if backup is not None:
            cache_file.write_bytes(backup)
        else:
            if cache_file.exists():
                cache_file.unlink()
