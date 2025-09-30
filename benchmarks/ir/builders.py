from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .gates import Gate


def hardware_efficient_ansatz(n_qubits: int, layers: int, start_index: int = 0) -> Tuple[List[Gate], int]:
    gates: List[Gate] = []
    idx = start_index
    for _ in range(layers):
        for wire in range(n_qubits):
            gates.append(Gate("RX", (wire,), idx))
            idx += 1
            gates.append(Gate("RY", (wire,), idx))
            idx += 1
        for wire in range(n_qubits):
            target = (wire + 1) % n_qubits
            if wire == target:
                continue
            gates.append(Gate("CNOT", (wire, target)))
    return gates, idx


def qaoa_layers_ir(
    edges: Sequence[Tuple[int, int]],
    p: int,
    n_qubits: int,
    start_index: int = 0,
) -> Tuple[List[Gate], int]:
    gates: List[Gate] = []
    idx = start_index
    for _ in range(p):
        gamma_index = idx
        idx += 1
        for a, b in edges:
            gates.append(Gate("CNOT", (a, b)))
            gates.append(Gate("RZ", (b,), gamma_index))
            gates.append(Gate("CNOT", (a, b)))
        beta_index = idx
        idx += 1
        for wire in range(n_qubits):
            gates.append(Gate("RX", (wire,), beta_index))
    return gates, idx


def angle_embedding_ir(num_features: int, start_index: int = 0) -> Tuple[List[Gate], int]:
    gates: List[Gate] = []
    idx = start_index
    for wire in range(num_features):
        gates.append(Gate("RY", (wire,), idx))
        idx += 1
    return gates, idx
