"""FakeRecogna 2.0: Detecção de Fake News em Português Brasileiro.

Pipeline modular pra reproduzir o experimento de detecção de fake news em
PT-BR usando o dataset FakeRecogna 2.0. Cada subpacote corresponde a uma
etapa do pipeline:

- `data`           -> carregamento, integridade e splits
- `preprocessing`  -> limpeza, sumarização e equalizador linguístico
- `features`       -> embeddings BERTimbau, TF-IDF e dataloaders
- `models`         -> baselines, redes neurais, ensembles, BERTimbau FT, PLMs
- `evaluation`     -> métricas, bootstrap, calibração, CV, stress tests, OOD
- `statistics`     -> log-odds (Monroe), McNemar com correção Holm
- `explainability` -> LIME, Integrated Gradients, Attention Rollout, estabilidade
- `adversarial`    -> perturbações de texto e back-translation
- `deployment`     -> benchmarks de latência e tamanho em disco
- `reports`        -> relatório consolidado em Markdown e JSON
- `utils`          -> seeds, logging, IO e utilitários de memória

Estado dinâmico (df, tokenizer, embeddings, ...) circula entre etapas pelo
`ExperimentContext`.
"""

import sys as _sys


def _ensure_utf8_stdio() -> None:
    """Reconfigura stdout/stderr pra UTF-8 (Windows PowerShell usa cp1252 por
    padrão, que quebra ao imprimir caracteres como ≥, Δ, →, emojis, etc.).

    No-op em sistemas onde a saída já é UTF-8 (Linux/macOS) ou onde os streams
    não são reconfiguráveis (já redirecionados pra um buffer binário).
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(_sys, stream_name, None)
        if stream is None:
            continue
        enc = getattr(stream, "encoding", None) or ""
        if enc.lower().replace("-", "") == "utf8":
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_ensure_utf8_stdio()

from ._context import ExperimentContext  # noqa: E402

__version__ = "0.9.0"

__all__ = ["ExperimentContext", "__version__"]
