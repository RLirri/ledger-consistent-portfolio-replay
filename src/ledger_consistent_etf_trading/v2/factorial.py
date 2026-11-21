"""Two-level full-factorial effects and the reconstruction identity used to validate E2 and E2b.

Cells are keyed by tuples of 0/1 levels. The decomposition is the usual one with -1/+1 coding:
    y(levels) = sum over subsets S of beta_S * prod_{k in S} s_k,   s_k = 2*level_k - 1
beta_() is the grand mean; beta_{k} is half the classical main effect. Nothing here assumes
monotonicity or that extreme cells bracket the others.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np


def effects(cells: dict, k: int) -> dict:
    keys = list(product((0, 1), repeat=k))
    if set(cells) != set(keys):
        raise ValueError("incomplete or mismatched factorial: every 0/1 combination is required exactly once")
    out = {}
    for r in range(k + 1):
        for S in combinations(range(k), r):
            out[S] = float(np.mean([cells[c] * np.prod([2 * c[j] - 1 for j in S]) for c in keys]))
    return out


def reconstruct(beta: dict, levels: tuple) -> float:
    return float(sum(b * np.prod([2 * levels[j] - 1 for j in S]) for S, b in beta.items()))


def reconstruction_error(cells: dict, k: int) -> float:
    beta = effects(cells, k)
    return max(abs(reconstruct(beta, c) - v) for c, v in cells.items())


def interaction_2x2(y00, y10, y01, y11) -> float:
    """I = Y(1,1) - Y(1,0) - Y(0,1) + Y(0,0)."""
    return y11 - y10 - y01 + y00
