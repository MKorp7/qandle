from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def state_fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> float:
    inner = torch.dot(torch.conj(state_a), state_b)
    return float(torch.abs(inner) ** 2)


def energy_expectation(state: torch.Tensor, hamiltonian: np.ndarray) -> torch.Tensor:
    h_tensor = torch.from_numpy(hamiltonian).to(state.dtype)
    if state.device.type != "cpu":
        h_tensor = h_tensor.to(state.device)
    energy = torch.conj(state).unsqueeze(0) @ (h_tensor @ state.unsqueeze(-1))
    return energy.squeeze()


def expected_maxcut_from_state(state: torch.Tensor, edges: Sequence[Tuple[int, int]], n_qubits: int) -> torch.Tensor:
    probs = torch.abs(state) ** 2
    total = torch.zeros((), dtype=probs.dtype, device=probs.device)
    for idx, prob in enumerate(probs):
        for a, b in edges:
            bit_a = (idx >> (n_qubits - a - 1)) & 1
            bit_b = (idx >> (n_qubits - b - 1)) & 1
            if bit_a != bit_b:
                total = total + prob
    return total
