import math
import time

import torch

import qandle.operators as op
from qandle.noise.channels import PhaseFlip
from qandle.qcircuit import Circuit


_PAULI_MATS = {
    "I": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128),
    "X": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128),
    "Y": torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128),
    "Z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128),
}


def _random_pauli_terms(n_qubits: int, n_terms: int, rng: torch.Generator) -> list[tuple[float, str]]:
    labels = "IXYZ"
    terms: list[tuple[float, str]] = []
    for _ in range(n_terms):
        chars = [labels[int(torch.randint(0, 4, (), generator=rng))] for _ in range(n_qubits)]
        string = "".join(chars)
        coeff = float(torch.randn((), generator=rng))
        terms.append((coeff, string))
    return terms


def _pauli_matrix(pauli: str) -> torch.Tensor:
    mat = torch.tensor([[1.0]], dtype=torch.complex128)
    for char in pauli:
        mat = torch.kron(mat, _PAULI_MATS[char])
    return mat


def _pauli_sum_expectation(rho: torch.Tensor, terms: list[tuple[float, str]]) -> float:
    total = 0.0
    for coeff, pauli in terms:
        mat = _pauli_matrix(pauli).to(rho.device, dtype=rho.dtype)
        total += coeff * torch.trace(rho @ mat).real.item()
    return total


def _build_noise_circuit(n_qubits: int, depth: int, rng: torch.Generator) -> Circuit:
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        for qubit in range(n_qubits):
            theta = float(2 * math.pi * torch.rand((), generator=rng))
            layers.append(op.RZ(qubit=qubit, theta=theta).build(n_qubits))
            layers.append(PhaseFlip(0.05, qubit).build(n_qubits))
        for qubit in range(0, n_qubits - 1, 2):
            layers.append(op.CNOT(qubit, qubit + 1).build(n_qubits))
            layers.append(PhaseFlip(0.02, qubit).build(n_qubits))
            layers.append(PhaseFlip(0.02, qubit + 1).build(n_qubits))
        for qubit in range(1, n_qubits - 1, 2):
            layers.append(op.CZ(qubit, qubit + 1).build(n_qubits))
            layers.append(PhaseFlip(0.02, qubit).build(n_qubits))
            layers.append(PhaseFlip(0.02, qubit + 1).build(n_qubits))
    return Circuit(layers=layers, num_qubits=n_qubits)


def _build_noise_heavy_circuit(n_qubits: int, depth: int, rng: torch.Generator) -> Circuit:
    layers: list[torch.nn.Module] = []
    for qubit in range(n_qubits):
        layers.append(op.RZ(qubit=qubit, theta=0.0).build(n_qubits))
    for _ in range(depth):
        for qubit in range(n_qubits):
            layers.append(PhaseFlip(0.1, qubit).build(n_qubits))
    return Circuit(layers=layers, num_qubits=n_qubits)


@torch.no_grad()
def test_ptm_matches_density_matrix_expectations():
    generator = torch.Generator().manual_seed(0)
    for n_qubits in (1, 3, 5, 8):
        circuit = _build_noise_circuit(n_qubits, depth=2, rng=generator)
        ptm_backend = circuit.forward(backend="ptm")
        dm_backend = circuit.forward(backend="density_matrix")
        terms = _random_pauli_terms(n_qubits, n_terms=12, rng=generator)
        ptm_val = ptm_backend.expectation_pauli_sum(terms)
        dm_val = _pauli_sum_expectation(dm_backend.rho, terms)
        assert abs(ptm_val - dm_val) <= 1e-4


@torch.no_grad()
def test_ptm_backend_is_faster_than_density_matrix():
    generator = torch.Generator().manual_seed(123)
    circuit = _build_noise_heavy_circuit(9, depth=16, rng=generator)

    # Warm up both backends.
    circuit.forward(backend="ptm")
    circuit.forward(backend="density_matrix")

    def _timed(name: str) -> float:
        start = time.perf_counter()
        circuit.forward(backend=name)
        return time.perf_counter() - start

    repeats = 1
    ptm_time = min(_timed("ptm") for _ in range(repeats))
    dm_time = min(_timed("density_matrix") for _ in range(repeats))
    assert dm_time / ptm_time >= 3.0
