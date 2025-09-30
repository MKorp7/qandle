import pytest


def test_controlled_gate_target_qubit_inference():
    torch = pytest.importorskip("torch")

    import qandle

    circuit = qandle.Circuit(
        [
            qandle.Controlled(0, qandle.RX(qubit=1, theta=0.123)),
        ]
    )

    state = torch.nn.functional.one_hot(
        torch.tensor(0, dtype=torch.long), num_classes=4
    ).to(torch.complex64)

    result = circuit(state)

    assert result.shape == state.shape
    assert circuit.num_qubits == 2
