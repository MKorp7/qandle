"""Control-gate statevector kernels."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["apply_CNOT", "apply_CCNOT"]


def _canonicalize_state(state: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if state.dim() == 1:
        return state.unsqueeze(0), True
    if state.dim() == 2:
        return state, False
    raise ValueError("state must be a 1D or 2D tensor of amplitudes")


def _infer_qubits(state: torch.Tensor, n_qubits: Optional[int]) -> int:
    if n_qubits is not None:
        return n_qubits
    size = state.shape[-1]
    if size == 0 or size & (size - 1):
        raise ValueError("state must represent a power-of-two sized vector")
    return size.bit_length() - 1


def _swap_amplitudes(
    reference: torch.Tensor,
    write_view: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
) -> None:
    if indices_a.numel() == 0:
        return
    write_view[:, indices_a] = reference[:, indices_b]
    write_view[:, indices_b] = reference[:, indices_a]


def apply_CNOT(
    state: torch.Tensor,
    control: int,
    target: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """Apply a CNOT gate to ``state`` without materialising the dense matrix."""

    if control == target:
        raise ValueError("control and target qubits must differ")

    working, squeeze = _canonicalize_state(state)
    n_qubits = _infer_qubits(working, n_qubits)
    dim = working.shape[-1]
    indices = torch.arange(dim, device=state.device)

    control_mask = 1 << control
    target_mask = 1 << target

    controlled = (indices & control_mask) != 0
    target_zero = (indices & target_mask) == 0
    swap_sources = torch.nonzero(controlled & target_zero, as_tuple=False).squeeze(-1)
    swap_targets = swap_sources ^ target_mask

    out = state.clone()
    out_view, _ = _canonicalize_state(out)
    _swap_amplitudes(working, out_view, swap_sources, swap_targets)
    if squeeze:
        out = out_view.squeeze(0)
    return out


def apply_CCNOT(
    state: torch.Tensor,
    control1: int,
    control2: int,
    target: int,
    n_qubits: Optional[int] = None,
) -> torch.Tensor:
    """Apply a CCNOT (Toffoli) gate to ``state`` without dense matrices."""

    if len({control1, control2, target}) != 3:
        raise ValueError("controls and target must all be distinct")

    working, squeeze = _canonicalize_state(state)
    n_qubits = _infer_qubits(working, n_qubits)
    dim = working.shape[-1]
    indices = torch.arange(dim, device=state.device)

    c1_mask = 1 << control1
    c2_mask = 1 << control2
    t_mask = 1 << target

    controls_on = ((indices & c1_mask) != 0) & ((indices & c2_mask) != 0)
    target_zero = (indices & t_mask) == 0
    swap_sources = torch.nonzero(controls_on & target_zero, as_tuple=False).squeeze(-1)
    swap_targets = swap_sources ^ t_mask

    out = state.clone()
    out_view, _ = _canonicalize_state(out)
    _swap_amplitudes(working, out_view, swap_sources, swap_targets)
    if squeeze:
        out = out_view.squeeze(0)
    return out
