import torch

import qandle


def test_controlled_gate_matches_backend_and_matrix():
    theta = torch.tensor(0.321, dtype=torch.float32)
    layers = [
        qandle.H(0, num_qubits=2),
        qandle.Controlled(0, qandle.RY(qubit=1, theta=theta)),
    ]
    circuit = qandle.Circuit(layers, num_qubits=2)

    backend = circuit(backend="statevector")
    backend_state = backend.state

    perm = []
    for idx in range(2 ** circuit.num_qubits):
        big_index = 0
        for qubit in range(circuit.num_qubits):
            bit = (idx >> qubit) & 1
            if bit:
                big_index |= 1 << (circuit.num_qubits - qubit - 1)
        perm.append(big_index)
    backend_state_big_endian = backend_state[perm]

    reference_state = circuit()

    assert torch.allclose(backend_state_big_endian, reference_state)

    matrix = circuit.to_matrix()
    initial_state = torch.zeros(2 ** circuit.num_qubits, dtype=reference_state.dtype)
    initial_state[0] = 1
    via_matrix = (matrix @ initial_state.reshape(-1, 1)).squeeze()

    assert torch.allclose(via_matrix, reference_state)
