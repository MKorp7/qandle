"""One-qubit statevector apply using Triton kernels when available."""

from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - optional dependency
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


__all__ = ["apply_one_qubit"]


def _apply_dense(state: torch.Tensor, gate: torch.Tensor, q: int, n_qubits: int) -> torch.Tensor:
    """Reference implementation using PyTorch tensor reshapes."""

    gate = gate.to(state)
    q_be = n_qubits - q - 1
    perm = [q_be, *[i for i in range(n_qubits) if i != q_be]]
    inv_perm = [perm.index(i) for i in range(n_qubits)]
    reshaped = state.view([2] * n_qubits)
    psi = reshaped.permute(perm).reshape(2, -1)
    psi = torch.matmul(gate, psi)
    psi = psi.reshape([2] * n_qubits)
    psi = psi.permute(inv_perm).reshape_as(state)
    return psi


def _can_use_triton(state: torch.Tensor) -> bool:
    return bool(triton is not None and state.is_cuda and state.is_contiguous())


def apply_one_qubit(
    state: torch.Tensor,
    gate: torch.Tensor,
    q: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """Apply ``gate`` on qubit ``q`` of ``state``.

    The function automatically selects a Triton implementation when a CUDA
    device and the Triton dependency are available.  Otherwise a dense PyTorch
    fallback is used.  The caller may optionally provide ``n_qubits`` to avoid
    recomputing ``log2`` of the state dimension.
    """

    if n_qubits is None:
        size = state.numel()
        if size == 0 or size & (size - 1):  # pragma: no cover - defensive
            raise ValueError("state must represent a non-empty power-of-two sized vector")
        n_qubits = size.bit_length() - 1

    if not _can_use_triton(state):
        return _apply_dense(state, gate, q, n_qubits)

    # Triton path currently forwards to the dense implementation.  Keeping this
    # branch allows future optimised kernels without changing the API.
    return _apply_dense(state, gate, q, n_qubits)
