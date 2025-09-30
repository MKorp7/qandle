from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple


def maxcut_value(bitstring: Sequence[int], edges: Sequence[Tuple[int, int]]) -> int:
    value = 0
    for a, b in edges:
        if bitstring[a] != bitstring[b]:
            value += 1
    return value


def optimal_maxcut(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> int:
    best = 0
    for bits in itertools.product([0, 1], repeat=n_qubits):
        value = maxcut_value(bits, edges)
        if value > best:
            best = value
    return best
