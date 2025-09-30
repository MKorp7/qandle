import math
import random
from typing import Iterable, List

import pytest
import torch

import qandle
from qandle import operators as op
from qandle.splitter import main as splitter_main


def _random_state(num_qubits: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    vec = torch.randn(2**num_qubits, dtype=torch.complex64)
    return vec / torch.linalg.norm(vec)


def _baseline_partition(layers: Iterable[qandle.operators.Operator], max_qubits: int) -> List[List[int]]:
    partitions: List[List[int]] = []
    current: List[int] = []
    current_qubits: set[int] = set()
    for idx, gate in enumerate(layers):
        qubits = set(splitter_main._extract_gate_qubits(gate))
        if not qubits:
            continue
        if len(qubits) > max_qubits:
            raise AssertionError(
                f"Gate at index {idx} spans {len(qubits)} qubits, exceeding baseline limit {max_qubits}."
            )
        if current and len(current_qubits | qubits) > max_qubits:
            partitions.append(list(current))
            current = []
            current_qubits = set()
        current.append(idx)
        current_qubits |= qubits
    if current:
        partitions.append(list(current))
    return partitions


def _run_split_and_measure(circuit: qandle.Circuit, max_qubits: int) -> tuple[float, int, int]:
    decomposed = circuit.decompose().circuit
    baseline = _baseline_partition(decomposed.layers, max_qubits)
    split_result = circuit.split(max_qubits=max_qubits)
    state = _random_state(circuit.num_qubits, seed=42 + max_qubits)
    reference = circuit(state.clone())
    splitted = split_result(state.clone())
    error = torch.linalg.norm(reference - splitted).item()
    carrier = getattr(split_result, "circuit", split_result)
    subcircuit_module = getattr(carrier, "subcircuits", None)
    subc_count = len(subcircuit_module) if subcircuit_module is not None else 0
    return error, subc_count, len(baseline)


def _ccnot_heavy_circuit(num_qubits: int, depth: int, seed: int) -> qandle.Circuit:
    rng = random.Random(seed)
    layers = []
    for _ in range(depth):
        c1, c2, target = rng.sample(range(num_qubits), 3)
        layers.append(op.CCNOT(c1, c2, target))
        # Sprinkle single-qubit rotations to exercise interleaving.
        for qubit in (c1, c2, target):
            layers.append(qandle.RX(qubit))
    return qandle.Circuit(num_qubits=num_qubits, layers=layers)


def _isolated_qubit_circuit(num_qubits: int, depth: int, seed: int) -> qandle.Circuit:
    rng = random.Random(seed)
    entangling_qubits = list(range(num_qubits - 2)) if num_qubits > 3 else list(range(num_qubits - 1))
    isolated_qubits = [q for q in range(num_qubits) if q not in entangling_qubits]
    layers = []
    for step in range(depth):
        if len(entangling_qubits) >= 2:
            c, t = rng.sample(entangling_qubits, 2)
            layers.append(op.CNOT(c, t))
            layers.append(qandle.RY(c))
            layers.append(qandle.RZ(t))
        if isolated_qubits:
            iso = isolated_qubits[step % len(isolated_qubits)]
            layers.append(qandle.RX(iso))
            layers.append(qandle.RZ(iso))
    return qandle.Circuit(num_qubits=num_qubits, layers=layers)


def _random_layered_circuit(num_qubits: int, depth: int, seed: int) -> qandle.Circuit:
    rng = random.Random(seed)
    entangling_builders = [
        lambda a, b: op.CNOT(a, b),
        lambda a, b: op.CZ(a, b),
        lambda a, b: op.SWAP(a, b),
    ]
    layers = []
    for _ in range(depth):
        if num_qubits >= 3 and rng.random() < 0.3:
            c1, c2, target = rng.sample(range(num_qubits), 3)
            layers.append(op.CCNOT(c1, c2, target))
            targets = (c1, c2, target)
        else:
            a, b = rng.sample(range(num_qubits), 2)
            layers.append(rng.choice(entangling_builders)(a, b))
            targets = (a, b)
        for qubit in targets:
            layers.append(qandle.RY(qubit))
    return qandle.Circuit(num_qubits=num_qubits, layers=layers)


@pytest.mark.parametrize("depth", [4, 6])
def test_ccnot_heavy_splitter(max_qubits: int, depth: int) -> None:
    num_qubits = max(max_qubits + 1, 6)
    circuit = _ccnot_heavy_circuit(num_qubits, depth, seed=depth * 31)
    error, subc_count, baseline = _run_split_and_measure(circuit, max_qubits)
    # Float32 arithmetic introduces ~1e-7 level noise; use a slightly looser
    # tolerance to avoid false failures while still enforcing near-identical states.
    assert error < 5e-7
    if baseline:
        assert subc_count <= math.ceil(1.5 * baseline)


@pytest.mark.parametrize("depth", [5, 7])
def test_isolated_qubit_handling(max_qubits: int, depth: int) -> None:
    num_qubits = max(max_qubits, 5)
    circuit = _isolated_qubit_circuit(num_qubits, depth, seed=depth * 17)
    error, subc_count, baseline = _run_split_and_measure(circuit, max_qubits)
    # Float32 arithmetic introduces ~1e-7 level noise; use a slightly looser
    # tolerance to avoid false failures while still enforcing near-identical states.
    assert error < 5e-7
    if baseline:
        assert subc_count <= math.ceil(1.5 * baseline)


@pytest.mark.parametrize("depth", [6, 8])
def test_random_layered_circuits(max_qubits: int, depth: int) -> None:
    num_qubits = max(max_qubits + 1, 5)
    circuit = _random_layered_circuit(num_qubits, depth, seed=depth * 23)
    error, subc_count, baseline = _run_split_and_measure(circuit, max_qubits)
    # Float32 arithmetic introduces ~1e-7 level noise; use a slightly looser
    # tolerance to avoid false failures while still enforcing near-identical states.
    assert error < 5e-7
    if baseline:
        assert subc_count <= math.ceil(1.5 * baseline)
