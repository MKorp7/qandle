from __future__ import annotations

from typing import Protocol

import torch

from benchmarks.ir.gates import Gate


class Backend(Protocol):
    name: str

    def simulate_state(
        self, n_qubits: int, gates: list[Gate], params: torch.Tensor, seed: int
    ) -> torch.Tensor: ...
