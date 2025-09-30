from __future__ import annotations

import torch

from benchmarks.backends.utils import apply_gate_matrix
from benchmarks.ir.gates import Gate


class QandleOriginBackend:
    name = "qadle_origin"

    def simulate_state(
        self, n_qubits: int, gates: list[Gate], params: torch.Tensor, seed: int
    ) -> torch.Tensor:
        import qandle.backends.statevector as statevector

        params = params.to(dtype=torch.float64)
        backend = statevector.StateVectorBackend(n_qubits=n_qubits, dtype=torch.complex128, device="cpu")
        dtype = torch.complex128
        for gate in gates:
            apply_gate_matrix(backend, gate, params, dtype)
        return backend.state.clone().to(torch.complex128)
