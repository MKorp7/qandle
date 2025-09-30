"""Monte-Carlo trajectory simulator using Pauli sampling."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..noise.model import NoiseModel


class TrajectoryBackend:
    """Stochastic trajectory simulator skeleton."""

    def __init__(self, num_qubits: int, noise_model: Optional[NoiseModel] = None, shots: int = 1024):
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        self.shots = shots
        self.state = torch.zeros(2 ** num_qubits, dtype=torch.cfloat)
        self.state[0] = 1.0

    def run(self, circuit: Any) -> dict[str, torch.Tensor]:
        """Execute ``shots`` noisy trajectories of ``circuit``.

        The circuit is evolved layer-by-layer on a state-vector backend.
        Noise channels inserted via the optional :class:`NoiseModel` are
        applied in their stochastic trajectory mode.  The final state of each
        trajectory is measured in the computational basis and aggregated into
        samples, counts and empirical probabilities.
        """

        from qandle.backends import StateVectorBackend  # local import
        from qandle.measurements import BuiltMeasurement  # local import
        from qandle.qcircuit import Circuit as CircuitCls  # local import
        from qandle.qcircuit import UnsplittedCircuit, _apply_backend_layer

        if self.shots < 0:
            raise ValueError("shots must be non-negative")

        if self.noise_model is not None and not isinstance(circuit, CircuitCls):
            raise TypeError("circuit must be a qandle.qcircuit.Circuit when using a noise model")

        working_circuit = circuit
        if self.noise_model is not None:
            working_circuit = self.noise_model.apply(circuit)

        if isinstance(working_circuit, CircuitCls):
            impl = working_circuit.circuit
            num_qubits = working_circuit.num_qubits
        elif isinstance(working_circuit, UnsplittedCircuit):
            impl = working_circuit
            num_qubits = working_circuit.num_qubits
        else:  # pragma: no cover - defensive programming
            raise TypeError("Unsupported circuit type for trajectory simulation")

        if not isinstance(impl, UnsplittedCircuit):
            impl = impl.decompose()
            num_qubits = impl.num_qubits

        if num_qubits != self.num_qubits:
            raise ValueError("Circuit qubit count does not match backend")

        layers = list(impl.layers)
        if any(isinstance(layer, BuiltMeasurement) for layer in layers):
            raise NotImplementedError("TrajectoryBackend does not support built measurement operations")

        backend = StateVectorBackend(self.num_qubits, dtype=self.state.dtype, device=self.state.device)
        initial_state = self.state.to(dtype=backend.state.dtype, device=backend.state.device).clone()
        generator = torch.Generator(device=backend.state.device)

        counts = torch.zeros(1 << self.num_qubits, dtype=torch.int64)
        samples = torch.zeros(self.shots, dtype=torch.int64)
        bitstrings = torch.zeros(self.shots, self.num_qubits, dtype=torch.int64)

        noise_kwargs = {"trajectory": True, "rng": generator}

        for shot in range(self.shots):
            backend.state = initial_state.clone()
            for layer in layers:
                _apply_backend_layer(layer, backend, noise_model=self.noise_model, noise_kwargs=noise_kwargs)

            probs = backend.measure()
            if probs.dim() != 1:
                raise RuntimeError("Measurement probabilities must be a 1D tensor")
            if probs.numel() != (1 << self.num_qubits):
                raise RuntimeError("Measurement probabilities have unexpected length")

            outcome = torch.multinomial(probs, num_samples=1, generator=generator).item()
            counts[outcome] += 1
            samples[shot] = outcome
            for idx, qubit in enumerate(range(self.num_qubits)):
                bitstrings[shot, idx] = (outcome >> (self.num_qubits - qubit - 1)) & 1

        if self.shots == 0:
            probabilities = counts.to(dtype=torch.float32)
        else:
            probabilities = counts.to(dtype=torch.float32) / float(self.shots)

        return {
            "bitstrings": bitstrings,
            "samples": samples,
            "counts": counts,
            "probabilities": probabilities,
        }
