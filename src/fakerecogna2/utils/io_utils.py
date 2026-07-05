"""IO de artefatos (tabelas, plots) e estruturas globais de resultados.

Expõe `save_table`, `save_plot`, `cleanup`, `RESULTS` e `RESULTS_MULTISEED`
redirecionando para `outputs/`.

`RESULTS` e `RESULTS_MULTISEED` são *dicts auto-persistentes*: cada escrita
grava em `outputs/.cache/results*.json`, e na próxima importação o conteúdo
é recarregado. Isso permite rodar etapas isoladas (`03_train_baselines.py`
seguido de `10_plm_finetune.py` em outro processo) e ainda assim ter
`16_final_results.csv` consolidado com todos os modelos.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:  # pragma: no cover
    _HAS_NP = False

from ..config import ARTIFACTS_DIR, TABLES_DIR, PLOTS_DIR, ensure_dirs
from .logging_utils import get_logger

ensure_dirs()


def _json_default(o: Any) -> Any:
    """Converte tipos não-JSON (numpy floats/ints, Paths, ...) para serialização."""
    if _HAS_NP:
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class _PersistentDict(dict):
    """dict que auto-persiste em JSON a cada mutação e recarrega na import.

    - Compatível com `dict` (subclasse direta), então todos os consumers
      existentes (que iteram, fazem `dict(R)`, `pd.DataFrame(R)`, etc.)
      continuam funcionando sem alteração.
    - Carga preguiçosa: lê do disco na primeira mutação ou leitura efetiva.
    - Variável de ambiente `FAKERECOGNA_RESULTS_FRESH=1` desabilita a leitura
      do cache, útil para iniciar uma execução do zero.
    """

    def __init__(self, persist_path: Path) -> None:
        super().__init__()
        self._persist_path = persist_path
        self._loaded = False
        # Tenta carregar de cara para que primeiros gets já vejam o conteúdo.
        self._load_if_needed()

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # marca antes de qualquer return cedo
        if os.environ.get("FAKERECOGNA_RESULTS_FRESH") == "1":
            return
        if not self._persist_path.exists():
            return
        try:
            with self._persist_path.open(encoding="utf-8") as f:
                data = json.load(f)
            # `dict.update` sem disparar nosso _save().
            super().update(data)
        except (OSError, json.JSONDecodeError) as e:
            get_logger().warning(
                f"[RESULTS] falha ao carregar {self._persist_path}: {e}"
            )

    def _save(self) -> None:
        import time

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(self), ensure_ascii=False, indent=2, default=_json_default)
        # Tenta rename atômico via .tmp; se Windows reclamar (anti-virus,
        # file watcher), retry curto e por fim escrita direta.
        tmp = self._persist_path.with_suffix(".json.tmp")
        for attempt in range(3):
            try:
                with tmp.open("w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, self._persist_path)
                return
            except OSError:
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
                    continue
        # Fallback: escrita direta (perde atomicidade mas evita travamento).
        try:
            with self._persist_path.open("w", encoding="utf-8") as f:
                f.write(payload)
        except OSError as e:
            get_logger().warning(f"[RESULTS] falha ao persistir: {e}")

    def __setitem__(self, key: Any, value: Any) -> None:
        self._load_if_needed()
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        self._load_if_needed()
        super().update(*args, **kwargs)
        self._save()

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._save()

    def pop(self, key: Any, *args, **kwargs):  # type: ignore[override]
        v = super().pop(key, *args, **kwargs)
        self._save()
        return v


# Caches ficam em outputs/.cache/ (escondidos do diretório de métricas
# para não poluir o que é entregue ao leitor).
_RESULTS_CACHE_DIR = ARTIFACTS_DIR / ".cache"
_RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Dicts globais persistentes para consolidação entre etapas.
RESULTS: _PersistentDict = _PersistentDict(_RESULTS_CACHE_DIR / "results.json")
RESULTS_MULTISEED: _PersistentDict = _PersistentDict(
    _RESULTS_CACHE_DIR / "results_multiseed.json"
)


def clear_results(also_multiseed: bool = True) -> None:
    """Limpa os dicts persistentes (memória + cache em disco).

    Útil antes de iniciar uma execução do zero quando hiperparâmetros mudaram
    e os resultados antigos no cache ficaram obsoletos.
    """
    RESULTS.clear()
    if also_multiseed:
        RESULTS_MULTISEED.clear()
    get_logger().info(
        f"[RESULTS] caches limpos (multiseed={'sim' if also_multiseed else 'nao'})"
    )


def cleanup() -> None:
    """Libera memória CPU/GPU."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_table(df: pd.DataFrame, name: str, subdir: str | None = None) -> Path:
    """Salva DataFrame como CSV em outputs/metrics (ou subdiretório).

    `name` pode conter separador de path (ex.: ``"lime/lime_00"``); subdirs
    intermediários são criados automaticamente.
    """
    base = TABLES_DIR if subdir is None else TABLES_DIR / subdir
    p = base / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    get_logger().info(f"Tabela salva: {p.name}")
    return p


def save_plot(fig, name: str, subdir: str | None = None) -> Path:
    """Salva figura matplotlib em outputs/figures.

    `name` pode conter separador de path (ex.: ``"lime/lime_00_TP_0"``);
    subdirs intermediários são criados automaticamente.
    """
    import matplotlib.pyplot as plt

    base = PLOTS_DIR if subdir is None else PLOTS_DIR / subdir
    p = base / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=150)
    get_logger().info(f"Plot salvo: {p.name}")
    plt.close(fig)
    return p


def save_json(obj: Any, name: str, subdir: str | None = None) -> Path:
    """Persiste objetos serializáveis (relatórios JSON, configs).

    `name` pode conter separador de path; subdirs são criados automaticamente.
    """
    base = TABLES_DIR if subdir is None else TABLES_DIR / subdir
    p = base / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    get_logger().info(f"JSON salvo: {p.name}")
    return p


__all__ = [
    "RESULTS",
    "RESULTS_MULTISEED",
    "clear_results",
    "cleanup",
    "save_table",
    "save_plot",
    "save_json",
    "ARTIFACTS_DIR",
]
