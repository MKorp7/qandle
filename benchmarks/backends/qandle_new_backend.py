from __future__ import annotations

import importlib
import os
import torch

from benchmarks.backends.utils import apply_gate_matrix
from benchmarks.ir.gates import Gate


class QandleNewBackend:
    name = "qandle_new"

    def __init__(self) -> None:
        qandle_path = os.environ.get("QANDLE_NEW_PATH")
        if not qandle_path:
            raise RuntimeError("QANDLE_NEW_PATH must be set before importing Qandle 2.0 backend")
        try:
            self._statevector_mod = importlib.import_module("qandle.backends.statevector")
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration benchmark runs
            if exc.name and exc.name.startswith("qandle"):
                raise RuntimeError(
                    "qandle.backends.statevector could not be imported. "
                    "Ensure QANDLE_NEW_PATH points to a Qandle 2.x checkout or install "
                    "qandle>=2.0 in the current environment."
                ) from exc
            raise

    def simulate_state(
        self, n_qubits: int, gates: list[Gate], params: torch.Tensor, seed: int
    ) -> torch.Tensor:
        params = params.to(dtype=torch.float64)
        backend = self._statevector_mod.StateVectorBackend(
            n_qubits=n_qubits, dtype=torch.complex128, device="cpu"
        )
        dtype = torch.complex128
        for gate in gates:
            apply_gate_matrix(backend, gate, params, dtype)
        return backend.state.clone().to(torch.complex128)
