import pytest
import torch


def _skip_if_no_compile(torch):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch version")


def _rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    half = theta.to(torch.float32) / 2
    cos = torch.cos(half).to(torch.complex64)
    sin = torch.sin(half).to(torch.complex64)
    row0 = torch.stack((cos, (-1j) * sin))
    row1 = torch.stack(((-1j) * sin, cos))
    return torch.stack((row0, row1))


def _ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    half = theta.to(torch.float32) / 2
    cos = torch.cos(half).to(torch.complex64)
    sin = torch.sin(half).to(torch.complex64)
    row0 = torch.stack((cos, -sin))
    row1 = torch.stack((sin, cos))
    return torch.stack((row0, row1))


def test_circuit_compile_context():
    torch = pytest.importorskip("torch")
    _skip_if_no_compile(torch)

    import qandle

    circuit = qandle.Circuit(
        [
            qandle.RX(qubit=0),
            qandle.RY(qubit=0),
            qandle.RZ(qubit=0),
        ],
        num_qubits=1,
    )

    state = torch.nn.functional.one_hot(
        torch.tensor(0, dtype=torch.long), num_classes=2
    ).to(torch.complex64)

    with qandle.compile(backend="eager") as compiler:
        compiled = compiler(circuit)

    expected = circuit(state)
    actual = compiled(state)

    torch.testing.assert_close(actual, expected)


def test_statevector_backend_compile():
    torch = pytest.importorskip("torch")
    _skip_if_no_compile(torch)

    import qandle

    def run(theta: torch.Tensor) -> torch.Tensor:
        backend = qandle.backends.StateVectorBackend(1, dtype=torch.complex64)
        backend.apply_1q(_rx_matrix(theta), 0)
        return backend.measure().real

    compiled = torch.compile(run, backend="eager")

    theta = torch.tensor(0.321, dtype=torch.float64)
    torch.testing.assert_close(compiled(theta), run(theta))


def test_mps_backend_compile_measurement():
    torch = pytest.importorskip("torch")
    _skip_if_no_compile(torch)

    import qandle

    def run(theta: torch.Tensor) -> torch.Tensor:
        backend = qandle.backends.MPSBackend(2, dtype=torch.complex64)
        backend.apply_1q(_rx_matrix(theta), 0)
        backend.apply_1q(_ry_matrix(theta), 1)
        return backend.measure([0, 1])

    compiled = torch.compile(run, backend="eager")

    theta = torch.tensor(0.123, dtype=torch.float64)
    torch.testing.assert_close(compiled(theta), run(theta))
