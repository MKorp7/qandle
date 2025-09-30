import torch

from qandle.noise.channels import AmplitudeDamping
from qandle.noise.model import NoiseModel
from qandle.qcircuit import Circuit
from qandle.simulators.trajectory_backend import TrajectoryBackend
from qandle.utils_gates import H


def _make_hadamard_circuit(num_qubits: int = 1) -> Circuit:
    layers = [H(0, num_qubits=num_qubits)]
    return Circuit(layers=layers, num_qubits=num_qubits)


def test_trajectory_backend_matches_unitary_probabilities():
    shots = 20000
    backend = TrajectoryBackend(num_qubits=1, shots=shots)
    circuit = _make_hadamard_circuit()

    results = backend.run(circuit)

    probs = results["probabilities"]
    expected = torch.tensor([0.5, 0.5], dtype=probs.dtype)

    assert torch.isclose(probs.sum(), torch.tensor(1.0, dtype=probs.dtype), atol=1e-6)
    assert torch.allclose(probs, expected, atol=0.05)
    assert int(results["counts"].sum().item()) == shots
    assert results["bitstrings"].shape == (shots, 1)


def test_trajectory_backend_samples_simple_noise_model():
    gamma = 0.4
    shots = 20000
    circuit = _make_hadamard_circuit()
    noise_model = NoiseModel(global_noise=[AmplitudeDamping(gamma, 0)])
    backend = TrajectoryBackend(num_qubits=1, noise_model=noise_model, shots=shots)

    results = backend.run(circuit)

    probs = results["probabilities"]
    expected = torch.tensor([(1.0 + gamma) / 2.0, (1.0 - gamma) / 2.0], dtype=probs.dtype)

    assert torch.isclose(probs.sum(), torch.tensor(1.0, dtype=probs.dtype), atol=1e-6)
    assert torch.allclose(probs, expected, atol=0.05)
    assert int(results["counts"].sum().item()) == shots
