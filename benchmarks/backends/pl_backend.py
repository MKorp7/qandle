from __future__ import annotations

import torch

from benchmarks.ir.gates import Gate


class PennyLaneBackend:
    name = "pennylane"

    def simulate_state(
        self, n_qubits: int, gates: list[Gate], params: torch.Tensor, seed: int
    ) -> torch.Tensor:
        import pennylane as qml

        params = params.to(dtype=torch.float64)
        dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(theta):
            for gate in gates:
                angle = theta[gate.param_index] if gate.param_index is not None else None
                if gate.name == "RX":
                    qml.RX(angle, wires=gate.wires[0])
                elif gate.name == "RY":
                    qml.RY(angle, wires=gate.wires[0])
                elif gate.name == "RZ":
                    qml.RZ(angle, wires=gate.wires[0])
                elif gate.name == "H":
                    qml.Hadamard(wires=gate.wires[0])
                elif gate.name == "CNOT":
                    qml.CNOT(wires=gate.wires)
                elif gate.name == "CZ":
                    qml.CZ(wires=gate.wires)
                else:
                    raise ValueError(f"Unsupported gate {gate.name}")
            return qml.state()

        state = circuit(params)
        return state.to(torch.complex128)
