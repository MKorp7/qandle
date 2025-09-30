"""Example circuit showing how to simulate noise with Qandle."""

import torch

import qandle


def main() -> None:
    """Build and execute a noisy quantum circuit on the MPS backend."""
    circuit = qandle.Circuit(
        layers=[
            qandle.AngleEmbedding(num_qubits=2),
            qandle.RX(qubit=0),
            qandle.CNOT(control=0, target=1),
            qandle.BitFlipNoise(qubit=0, p=0.01),
            qandle.PhaseFlipNoise(qubit=1, p=0.02),
            qandle.DepolarizingNoise(qubit=1, p=0.05),
            qandle.MeasureProbability(),
        ],
        backend=qandle.backends.MPSBackend(max_bond_dim=32),
    )

    # Random complex input state
    input_state = torch.rand(circuit.num_qubits, dtype=torch.complex64)

    probabilities = circuit(input_state)
    print("Measurement probabilities:")
    print(probabilities)


if __name__ == "__main__":
    main()
