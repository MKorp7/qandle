import torch

from qandle.backends import DensityMatrixBackend
from qandle.noise.channels import AmplitudeDamping
from qandle.operators import CNOT
from qandle.qcircuit import Circuit
from qandle.utils_gates import H, X


def _outer(psi: torch.Tensor) -> torch.Tensor:
    return psi.unsqueeze(1) @ psi.conj().unsqueeze(0)


def test_density_matrix_matches_statevector():
    layers = [
        H(0, num_qubits=2),
        X(1, num_qubits=2),
        CNOT(0, 1).build(2),
    ]
    circuit = Circuit(layers=layers, num_qubits=2)

    sv_backend = circuit.forward(backend="statevector")
    dm_backend = circuit.forward(backend="density_matrix")

    psi = sv_backend.state
    rho = dm_backend.rho

    assert torch.allclose(rho, _outer(psi), atol=1e-6)
    assert torch.allclose(dm_backend.measure(), sv_backend.measure(), atol=1e-6)


def test_amplitude_damping_channel():
    backend = DensityMatrixBackend(1)
    x_gate = torch.tensor([[0, 1], [1, 0]], dtype=backend.rho.dtype, device=backend.rho.device)
    backend.apply_1q(x_gate, 0)

    gamma = 0.3
    channel = AmplitudeDamping(gamma, 0).build(1)
    backend.rho = channel(backend.rho)

    expected = torch.tensor(
        [[gamma, 0.0], [0.0, 1 - gamma]],
        dtype=backend.rho.dtype,
        device=backend.rho.device,
    )
    assert torch.allclose(backend.rho, expected, atol=1e-6)


def test_measurement_probabilities_and_sampling():
    backend = DensityMatrixBackend(1)
    probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
    backend.rho = torch.diag(probs.to(dtype=backend.rho.real.dtype)).to(
        dtype=backend.rho.dtype, device=backend.rho.device
    )

    measured = backend.measure()
    assert torch.allclose(measured, probs.to(device=backend.rho.device), atol=1e-6)

    generator = torch.Generator(device=backend.rho.device).manual_seed(0)
    outcomes = torch.multinomial(measured, 4000, replacement=True, generator=generator)
    freqs = torch.bincount(outcomes, minlength=2).float() / 4000
    assert torch.allclose(freqs, torch.tensor([0.5, 0.5]), atol=0.05)


def test_density_matrix_supports_autograd():
    backend = DensityMatrixBackend(1, dtype=torch.complex128)
    theta = torch.tensor(0.321, dtype=torch.float64, requires_grad=True)

    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    matrix = torch.stack(
        [
            torch.stack([cos, -1j * sin]),
            torch.stack([-1j * sin, cos]),
        ]
    ).to(dtype=backend.rho.dtype)

    backend.apply_1q(matrix, 0)
    z_op = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=backend.rho.dtype, device=backend.rho.device)
    expectation = torch.real(torch.trace(backend.rho @ z_op))
    expectation.backward()

    assert theta.grad is not None
    assert theta.grad.abs() > 0
