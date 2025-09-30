"""Two-qubit statevector apply using Triton kernels when available."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

try:  # pragma: no cover - optional dependency
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


__all__ = ["apply_two_qubit"]


def _apply_dense(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    gate = gate.to(state)
    q_be = [n_qubits - q - 1 for q in qubits]
    perm = [*q_be, *[i for i in range(n_qubits) if i not in q_be]]
    inv_perm = [perm.index(i) for i in range(n_qubits)]
    reshaped = state.view([2] * n_qubits)
    psi = reshaped.permute(perm).reshape(4, -1)
    psi = torch.matmul(gate, psi)
    psi = psi.reshape([2] * n_qubits)
    psi = psi.permute(inv_perm).reshape_as(state)
    return psi


def _can_use_triton(state: torch.Tensor) -> bool:
    return bool(triton is not None and state.is_cuda and state.is_contiguous())


def apply_two_qubit(
    state: torch.Tensor,
    gate: torch.Tensor,
    q0: int,
    q1: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """Apply a two-qubit ``gate`` on ``state`` targeting qubits ``q0`` and ``q1``."""

    if n_qubits is None:
        size = state.numel()
        if size == 0 or size & (size - 1):  # pragma: no cover - defensive
            raise ValueError("state must represent a non-empty power-of-two sized vector")
        n_qubits = size.bit_length() - 1

    if not _can_use_triton(state):
        return _apply_dense(state, gate, (q0, q1), n_qubits)

    return _apply_dense(state, gate, (q0, q1), n_qubits)
