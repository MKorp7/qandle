import pytest
import torch

from qandle.backends.stabilizer import StabilizerBackend
from qandle.operators import CNOT, CZ, SWAP, U
from qandle.qcircuit import Circuit
from qandle.utils_gates import H, S, T, X, Z


def _sdg_gate() -> torch.Tensor:
    return torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64)


def test_tableau_updates_single_and_two_qubit_gates():
    backend_h = StabilizerBackend(1)
    backend_h.apply_1q(H(0, num_qubits=1).to_matrix(), 0)
    _, phases = backend_h.tableau()
    assert backend_h.pauli_string(0) == "Z"
    assert backend_h.pauli_string(1) == "X"
    assert torch.all(phases == 0)

    backend_s = StabilizerBackend(1)
    backend_s.apply_1q(S(0, num_qubits=1).to_matrix(), 0)
    _, phases = backend_s.tableau()
    assert backend_s.pauli_string(0) == "Y"
    assert backend_s.pauli_string(1) == "Z"
    assert torch.all(phases == phases % 4)

    backend_cnot = StabilizerBackend(2)
    backend_cnot.apply_2q(CNOT(0, 1).build(2).to_matrix(), 0, 1)
    _, phases = backend_cnot.tableau()
    assert backend_cnot.pauli_string(0) == "XX"
    assert backend_cnot.pauli_string(1) == "IX"
    assert backend_cnot.pauli_string(2) == "ZI"
    assert backend_cnot.pauli_string(3) == "ZZ"
    assert torch.all(phases == phases % 4)


def _clifford_circuits():
    sdg = U(0, _sdg_gate())
    return [
        (2, [H(0, num_qubits=2), X(1, num_qubits=2), CNOT(0, 1).build(2)]),
        (
            2,
            [
                S(0, num_qubits=2),
                H(0, num_qubits=2),
                sdg.build(2),
                Z(1, num_qubits=2),
                CNOT(0, 1).build(2),
            ],
        ),
        (
            3,
            [
                H(0, num_qubits=3),
                H(1, num_qubits=3),
                CNOT(0, 1).build(3),
                CZ(1, 2).build(3),
                SWAP(0, 2).build(3),
                Z(2, num_qubits=3),
            ],
        ),
    ]


@pytest.mark.parametrize("num_qubits,layers", _clifford_circuits())
def test_stabilizer_matches_statevector(num_qubits, layers):
    circuit = Circuit(layers=layers, num_qubits=num_qubits)

    sv_backend = circuit.forward(backend="statevector")
    stabilizer_backend = circuit.forward(backend="stabilizer")

    assert torch.allclose(
        stabilizer_backend.to_statevector(), sv_backend.state, atol=1e-6
    )
    assert torch.allclose(
        stabilizer_backend.measure(), sv_backend.measure(), atol=1e-6
    )


def test_stabilizer_backend_rejects_non_clifford_gate():
    circuit = Circuit(layers=[T(0, num_qubits=1)], num_qubits=1)
    with pytest.raises(ValueError, match="Clifford"):
        circuit.forward(backend="stabilizer")


def test_forward_accepts_backend_instance():
    layers = [H(0, num_qubits=2), CNOT(0, 1).build(2)]
    circuit = Circuit(layers=layers, num_qubits=2)

    backend = StabilizerBackend(2)
    result_backend = circuit.forward(backend=backend)

    assert result_backend is backend

    sv_backend = circuit.forward(backend="statevector")
    assert torch.allclose(result_backend.to_statevector(), sv_backend.state, atol=1e-6)
