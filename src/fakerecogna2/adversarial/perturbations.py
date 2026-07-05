"""Perturbações de texto: typos, deletion, swap."""

from __future__ import annotations

import random

_KEYBOARD: dict[str, str] = {
    "a": "sqwz", "s": "adwe", "d": "sfre", "f": "dgre", "g": "fhty",
    "h": "gjtu", "j": "hkui", "k": "jlio", "l": "ko",
    "q": "wa", "w": "qesa", "e": "wrsd", "r": "etfd", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uokj", "o": "iplk",
}




# -- API limpa ----------------------------------------------------------------
def adv_random_typos(
    text: str, rate: float = 0.05, rng: random.Random | None = None
) -> str:
    """Introduz typos em ~rate das palavras: swap/sub/del/dup."""
    rng = rng or random.Random()
    words = text.split()
    n_perturb = max(1, int(len(words) * rate))
    idxs = rng.sample(range(len(words)), min(n_perturb, len(words)))
    for i in idxs:
        w = words[i]
        if len(w) < 3:
            continue
        op = rng.choice(["swap", "sub", "del", "dup"])
        p = rng.randint(1, len(w) - 2)
        if op == "swap" and p < len(w) - 1:
            words[i] = w[:p] + w[p + 1] + w[p] + w[p + 2 :]
        elif op == "sub" and w[p] in _KEYBOARD:
            words[i] = w[:p] + rng.choice(_KEYBOARD[w[p]]) + w[p + 1 :]
        elif op == "del":
            words[i] = w[:p] + w[p + 1 :]
        elif op == "dup":
            words[i] = w[:p] + w[p] + w[p:]
    return " ".join(words)


def adv_word_deletion(
    text: str, rate: float = 0.10, rng: random.Random | None = None
) -> str:
    """Remove ~rate das palavras aleatoriamente."""
    rng = rng or random.Random()
    words = text.split()
    keep = [w for w in words if rng.random() > rate]
    return " ".join(keep) if keep else text


def adv_word_swap(
    text: str, rate: float = 0.05, rng: random.Random | None = None
) -> str:
    """Troca a ordem de pares de palavras adjacentes em ~rate das posições."""
    rng = rng or random.Random()
    words = text.split()
    n_swaps = max(1, int(len(words) * rate))
    for _ in range(n_swaps):
        if len(words) < 2:
            break
        i = rng.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


__all__ = [
    "adv_random_typos",
    "adv_word_deletion",
    "adv_word_swap",
]
