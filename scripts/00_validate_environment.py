"""Valida o ambiente: imports, GPU e modelo spaCy PT.

Equivale às células 1.1 e 1.2 do notebook (instalação + setup determinístico).
"""

from __future__ import annotations

import sys
from importlib import import_module

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "statsmodels", "yaml",
    "matplotlib", "seaborn", "tabulate", "tqdm",
    "nltk", "spacy", "datasets", "datasketch", "sumy",
    "torch", "transformers", "lime", "captum",
]


def main() -> int:
    missing = []
    for name in REQUIRED:
        try:
            import_module(name)
            print(f"  OK   {name}")
        except ImportError as e:
            print(f"  FAIL {name} -> {e}")
            missing.append(name)

    try:
        import torch
        print(f"\nTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Torch check falhou: {e}")

    try:
        import spacy
        spacy.load("pt_core_news_sm")
        print("spaCy pt_core_news_sm: OK")
    except OSError:
        print("spaCy pt_core_news_sm AUSENTE — rode: python -m spacy download pt_core_news_sm")
        missing.append("pt_core_news_sm")

    if missing:
        print(f"\nDependências faltando: {missing}")
        return 1
    print("\nAmbiente OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
