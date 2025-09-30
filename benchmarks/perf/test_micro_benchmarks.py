"""Micro/meso benchmarks exercising common simulator paths."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import pytest
import torch

import qandle
from qandle.backends import MPSBackend, StateVectorBackend
from qandle.gradients import parameter_shift_forward
from qandle.operators import CNOT, RX, RY, RZ

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rotation_x(theta: float) -> torch.Tensor:
    half = theta / 2.0
    cos = math.cos(half)
    sin = math.sin(half)
    return torch.tensor(
        [[cos, -1j * sin], [-1j * sin, cos]],
        dtype=torch.complex64,
    )


def _rotation_y(theta: float) -> torch.Tensor:
    half = theta / 2.0
    cos = math.cos(half)
    sin = math.sin(half)
    return torch.tensor(
        [[cos, -sin], [sin, cos]],
        dtype=torch.complex64,
    )


def _rotation_z(theta: float) -> torch.Tensor:
    half = theta / 2.0
    return torch.tensor(
        [[cmath.exp(-1j * half), 0.0], [0.0, cmath.exp(1j * half)]],
        dtype=torch.complex64,
    )


_CNOT = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=torch.complex64,
)


@dataclass(frozen=True)
class GateSpec:
    kind: str
    qubits: tuple[int, ...]
    matrix: torch.Tensor


# ---------------------------------------------------------------------------
# Backend evolution benchmarks
# ---------------------------------------------------------------------------

_BACKEND_QUBITS = 10
_BACKEND_DEPTH = 4
_BACKEND_SEQUENCE: list[GateSpec] = []
_generator = torch.Generator().manual_seed(1234)
for layer in range(_BACKEND_DEPTH):
    angles = torch.rand((_BACKEND_QUBITS, 3), generator=_generator)
    for q in range(_BACKEND_QUBITS):
        # Deterministic single-qubit rotation bundle
        rx = _rotation_x(float(angles[q, 0] * math.pi))
        ry = _rotation_y(float(angles[q, 1] * math.pi))
        rz = _rotation_z(float(angles[q, 2] * math.pi))
        _BACKEND_SEQUENCE.append(GateSpec("1q", (q,), rx))
        _BACKEND_SEQUENCE.append(GateSpec("1q", (q,), ry))
        _BACKEND_SEQUENCE.append(GateSpec("1q", (q,), rz))
    start = layer % 2
    for q in range(start, _BACKEND_QUBITS - 1, 2):
        _BACKEND_SEQUENCE.append(GateSpec("2q", (q, q + 1), _CNOT))


@pytest.mark.benchmark(group="backend_evolution")
@pytest.mark.parametrize("backend_cls", [StateVectorBackend, MPSBackend])
def test_backend_time(backend_cls, benchmark):
    """Apply a fixed random circuit on both backends."""

    def _run() -> torch.Tensor:
        backend = backend_cls(n_qubits=_BACKEND_QUBITS)
        for spec in _BACKEND_SEQUENCE:
            if spec.kind == "1q":
                backend.apply_1q(spec.matrix, spec.qubits[0])
            else:
                backend.apply_2q(spec.matrix, spec.qubits[0], spec.qubits[1])
        state = getattr(backend, "state", None)
        return state if state is not None else torch.empty(0)

    benchmark(_run)


# ---------------------------------------------------------------------------
# Circuit splitting benchmarks
# ---------------------------------------------------------------------------

_CIRCUIT_QUBITS = 6
_CIRCUIT_DEPTH = 6
_layers: list[qandle.operators.Operator] = []
_layer_gen = torch.Generator().manual_seed(2024)
for depth in range(_CIRCUIT_DEPTH):
    thetas = torch.rand((_CIRCUIT_QUBITS, 2), generator=_layer_gen)
    for qubit in range(_CIRCUIT_QUBITS):
        _layers.append(RY(qubit, theta=float(thetas[qubit, 0]) * math.pi))
        _layers.append(RZ(qubit, theta=float(thetas[qubit, 1]) * math.pi))
    offset = depth % 2
    for control in range(offset, _CIRCUIT_QUBITS - 1, 2):
        _layers.append(CNOT(control, control + 1))

_BASE_CIRCUIT = qandle.Circuit(_layers, num_qubits=_CIRCUIT_QUBITS)
_UNSPLIT_IMPL = _BASE_CIRCUIT.circuit
_SPLIT_CIRCUIT = _BASE_CIRCUIT.split(max_qubits=3)
_SPLIT_IMPL = _SPLIT_CIRCUIT.circuit

_state_gen = torch.Generator().manual_seed(9001)
_state_real = torch.randn(2 ** _CIRCUIT_QUBITS, generator=_state_gen)
_state_imag = torch.randn(2 ** _CIRCUIT_QUBITS, generator=_state_gen)
_INPUT_STATE = torch.complex(_state_real, _state_imag)
_INPUT_STATE = _INPUT_STATE / _INPUT_STATE.norm()


@pytest.mark.benchmark(group="circuit_split")
def test_unsplit_forward(benchmark):
    def _run() -> torch.Tensor:
        return _UNSPLIT_IMPL.forward(_INPUT_STATE.clone())

    benchmark(_run)


@pytest.mark.benchmark(group="circuit_split")
def test_split_forward(benchmark):
    def _run() -> torch.Tensor:
        return _SPLIT_IMPL.forward(_INPUT_STATE.clone())

    benchmark(_run)


# ---------------------------------------------------------------------------
# Dense vs. sparse gate application benchmarks
# ---------------------------------------------------------------------------

_DENSE_MATRIX = _UNSPLIT_IMPL.to_matrix()
_SPARSE_SEQUENCE: list[GateSpec] = []
for layer in _UNSPLIT_IMPL.layers:
    if hasattr(layer, "qubit"):
        matrix = layer.to_matrix()
        _SPARSE_SEQUENCE.append(GateSpec("1q", (layer.qubit,), matrix))
    elif hasattr(layer, "c") and hasattr(layer, "t") and not hasattr(layer, "c2"):
        matrix = layer.to_matrix()
        _SPARSE_SEQUENCE.append(GateSpec("2q", (layer.c, layer.t), matrix))
    else:
        raise RuntimeError(f"Unsupported gate in sparse benchmark: {layer!r}")


@pytest.mark.benchmark(group="dense_vs_sparse")
def test_dense_gate_application(benchmark):
    def _run() -> torch.Tensor:
        backend = StateVectorBackend(n_qubits=_CIRCUIT_QUBITS)
        backend.state = _INPUT_STATE.clone()
        backend.apply_dense(_DENSE_MATRIX, tuple(range(_CIRCUIT_QUBITS)))
        return backend.state

    benchmark(_run)


@pytest.mark.benchmark(group="dense_vs_sparse")
def test_sparse_gate_application(benchmark):
    def _run() -> torch.Tensor:
        backend = StateVectorBackend(n_qubits=_CIRCUIT_QUBITS)
        backend.state = _INPUT_STATE.clone()
        for spec in _SPARSE_SEQUENCE:
            if spec.kind == "1q":
                backend.apply_1q(spec.matrix, spec.qubits[0])
            else:
                backend.apply_2q(spec.matrix, spec.qubits[0], spec.qubits[1])
        return backend.state

    benchmark(_run)


# ---------------------------------------------------------------------------
# Gradient benchmark
# ---------------------------------------------------------------------------


class _ExpectationModule(torch.nn.Module):
    def __init__(self, circuit: qandle.qcircuit.UnsplittedCircuit):
        super().__init__()
        self.circuit = circuit

    def forward(self, state: torch.Tensor | None = None) -> torch.Tensor:
        final_state = self.circuit.forward(state)
        view = final_state.view(2, -1)
        expect = view[0].abs().pow(2).sum() - view[1].abs().pow(2).sum()
        return expect.real


_GRAD_QUBITS = 4
_grad_layers: list[qandle.operators.Operator] = []
_grad_gen = torch.Generator().manual_seed(1337)
for depth in range(5):
    angles = torch.rand((_GRAD_QUBITS, 3), generator=_grad_gen)
    for qubit in range(_GRAD_QUBITS):
        _grad_layers.append(RX(qubit, theta=float(angles[qubit, 0]) * math.pi))
        _grad_layers.append(RY(qubit, theta=float(angles[qubit, 1]) * math.pi))
        _grad_layers.append(RZ(qubit, theta=float(angles[qubit, 2]) * math.pi))
    for control in range(depth % 2, _GRAD_QUBITS - 1, 2):
        _grad_layers.append(CNOT(control, control + 1))

_grad_circuit = qandle.Circuit(_grad_layers, num_qubits=_GRAD_QUBITS).circuit
_EXPECTATION = _ExpectationModule(_grad_circuit)
_basis_state = torch.nn.functional.one_hot(
    torch.tensor(0, dtype=torch.long),
    num_classes=2 ** _GRAD_QUBITS,
).to(dtype=torch.complex64)


@pytest.mark.benchmark(group="parameter_shift")
def test_parameter_shift_gradient(benchmark):
    def _run() -> torch.Tensor:
        _EXPECTATION.zero_grad(set_to_none=True)
        loss = parameter_shift_forward(_EXPECTATION, state=_basis_state)
        loss.backward()
        return loss

    benchmark(_run)


__all__ = [
    "test_backend_time",
    "test_unsplit_forward",
    "test_split_forward",
    "test_dense_gate_application",
    "test_sparse_gate_application",
    "test_parameter_shift_gradient",
]
