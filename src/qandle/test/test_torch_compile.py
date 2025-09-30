import pytest


def test_circuit_torch_compile():
    torch = pytest.importorskip("torch")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch version")

    import qandle

    circuit = qandle.Circuit(
        [
            qandle.RX(qubit=0),
            qandle.RY(qubit=0),
            qandle.RZ(qubit=0),
        ],
        num_qubits=1,
    )

    state = torch.zeros(2, dtype=torch.complex64)
    state[0] = 1

    compiled = torch.compile(circuit)

    expected = circuit(state)
    actual = compiled(state)

    assert torch.allclose(actual, expected)
