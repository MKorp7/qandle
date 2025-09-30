import math

import pytest

torch = pytest.importorskip("torch")

from qandle.operators import Controlled, RX, U
from qandle.qcircuit import Circuit


def _x_gate(qubit: int, num_qubits: int) -> torch.nn.Module:
    matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    return U(qubit=qubit, matrix=matrix).build(num_qubits=num_qubits)


def test_controlled_named_parameter_affects_statevector_backend():
    num_qubits = 2
    layers = [
        _x_gate(0, num_qubits),
        Controlled(0, RX(qubit=1, name="phi")),
    ]
    circuit = Circuit(layers=layers, num_qubits=num_qubits)

    backend_zero = circuit(backend="statevector", phi=0.0)
    backend_varied = circuit(backend="statevector", phi=math.pi / 2)

    state_zero = backend_zero.state.clone()
    state_varied = backend_varied.state.clone()

    assert not torch.allclose(state_zero, state_varied)
