from __future__ import annotations

import functools
from typing import Tuple

import numpy as np


_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

_TERMS: Tuple[Tuple[float, str], ...] = (
    (-1.052373245772859, "IIII"),
    (0.39793742484318045, "ZIII"),
    (-0.39793742484318045, "IZII"),
    (-0.01128010425623538, "IIZI"),
    (0.01128010425623538, "IIIZ"),
    (0.18093119978423156, "ZZII"),
    (0.18093119978423156, "IIZZ"),
    (0.1689275385461718, "ZIZI"),
    (0.12093265369994998, "ZIIZ"),
    (0.1689275385461718, "IZIZ"),
    (0.12093265369994998, "IZZI"),
    (0.04523279994605782, "XXYY"),
    (-0.04523279994605782, "XYXY"),
    (-0.04523279994605782, "YXXY"),
    (0.04523279994605782, "YYXX"),
)


@functools.lru_cache(maxsize=1)
def h2_hamiltonian() -> np.ndarray:
    matrices = []
    for coeff, paulis in _TERMS:
        term = _PAULI[paulis[0]]
        for char in paulis[1:]:
            term = np.kron(term, _PAULI[char])
        matrices.append(coeff * term)
    return np.sum(matrices, axis=0)


@functools.lru_cache(maxsize=1)
def h2_ground_energy(hamiltonian: np.ndarray | None = None) -> float:
    if hamiltonian is None:
        hamiltonian = h2_hamiltonian()
    eigvals = np.linalg.eigh(hamiltonian)[0]
    return float(np.min(eigvals).real)
