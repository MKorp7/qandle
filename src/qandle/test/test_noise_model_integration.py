import pytest

try:  # pragma: no cover - environment guard
    import torch

    from qandle import operators
    from qandle.noise import AmplitudeDamping, NoiseModel, PhaseFlip
    from qandle.qcircuit import Circuit
except ModuleNotFoundError:  # torch is not available in the environment
    pytestmark = pytest.mark.skip(reason="torch is required for noise integration tests")
else:


    def hadamard() -> torch.Tensor:
        return torch.tensor(
            [[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64
        ) / torch.sqrt(torch.tensor(2.0, dtype=torch.complex64))


    def plus_density() -> torch.Tensor:
        vec = torch.tensor([1.0, 1.0], dtype=torch.complex64) / torch.sqrt(
            torch.tensor(2.0, dtype=torch.complex64)
        )
        return torch.outer(vec, vec.conj())


    def test_global_noise_applied_after_each_gate():
        circuit = Circuit([operators.U(0, torch.eye(2, dtype=torch.complex64))], num_qubits=1)
        model = NoiseModel(global_noise=[PhaseFlip(0.2, 0)])

        out = circuit.forward(state=plus_density(), noise_model=model)

        expected_offdiag = torch.tensor(0.3, dtype=torch.float32)
        assert torch.allclose(out[0, 1].real, expected_offdiag, atol=1e-6)


    def test_gate_specific_noise_mapping():
        circuit = Circuit([operators.U(0, torch.eye(2, dtype=torch.complex64))], num_qubits=1)
        model = NoiseModel(channels={"U": PhaseFlip(0.5, 0)})

        out = circuit.forward(state=plus_density(), noise_model=model)
        expected = torch.tensor(
            [[0.5, 0.0], [0.0, 0.5]], dtype=torch.complex64
        )
        assert torch.allclose(out, expected, atol=1e-6)


    def test_noise_model_on_statevector_backend():
        circuit = Circuit([operators.U(0, hadamard())], num_qubits=1)
        model = NoiseModel(channels={"U": PhaseFlip(1.0, 0)})
        gen = torch.Generator().manual_seed(0)

        backend = circuit.forward(
            backend="statevector",
            noise_model=model,
            noise_kwargs={"trajectory": True, "rng": gen},
        )

        expected = torch.tensor([1.0, -1.0], dtype=torch.complex64) / torch.sqrt(
            torch.tensor(2.0, dtype=torch.complex64)
        )
        assert torch.allclose(backend.state, expected, atol=1e-6)


    def test_noise_model_sample_without_noise():
        circuit = Circuit([operators.U(0, hadamard())], num_qubits=1)
        model = NoiseModel()

        shots = 2000
        out = model.sample(circuit, n_shots=shots, seed=0)

        assert out["bitstrings"].shape == (shots, 1)
        assert out["counts"].sum() == shots
        probs = out["probabilities"].float()
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
        # Expect approximately uniform distribution for a Hadamard on |0>.
        assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=0.1)


    def test_noise_model_sample_with_amplitude_damping():
        X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
        circuit = Circuit([operators.U(0, X)], num_qubits=1)
        model = NoiseModel(channels={"U": AmplitudeDamping(1.0, 0)})

        out = model.sample(circuit, n_shots=16, seed=1)

        assert torch.all(out["bitstrings"] == 0)
        assert out["counts"][0] == 16


    def test_noise_model_sample_subset_qubits():
        had = hadamard()
        circuit = Circuit([operators.U(0, had)], num_qubits=2)
        model = NoiseModel()

        shots = 1024
        out = model.sample(circuit, n_shots=shots, measured_qubits=[1], seed=123)

        assert out["bitstrings"].shape == (shots, 1)
        assert out["counts"].numel() == 2
        assert torch.isclose(out["probabilities"].sum(), torch.tensor(1.0))
        assert torch.allclose(out["probabilities"], torch.tensor([0.5, 0.5]), atol=0.1)

