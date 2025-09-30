"""Density-matrix simulator backend with Kraus noise support."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from . import QuantumBackend
from ..noise.channels import _apply_kraus_density, _get_perm_cache


class DensityMatrixBackend(QuantumBackend):
    """Exact density-matrix simulator backend."""

    def __init__(
        self,
        n_qubits: int,
        *,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str = torch.device("cpu"),
        max_qubits: int | None = 10,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.max_qubits = max_qubits
        self._perm_cache: dict[tuple[int, ...], object] = {}
        self.allocate(n_qubits)

    # ------------------------------------------------------------------
    # State allocation utilities
    def allocate(
        self,
        n_qubits: int,
        *,
        state: torch.Tensor | None = None,
        rho: torch.Tensor | None = None,
    ):
        if self.max_qubits is not None and n_qubits > self.max_qubits:
            raise ValueError(
                "DensityMatrixBackend allocates a 2**n x 2**n density matrix. "
                "Set max_qubits=None explicitly to simulate more than "
                f"{self.max_qubits} qubits."
            )

        self.n_qubits = n_qubits
        self._perm_cache = {}
        dim = 1 << n_qubits

        if rho is not None:
            if rho.shape != (dim, dim):
                raise ValueError("rho must have shape (2**n, 2**n)")
            self.rho = rho.to(device=self.device, dtype=self.dtype)
            return self

        if state is not None:
            state = state.to(device=self.device, dtype=self.dtype)
            if state.dim() != 1 or state.shape[0] != dim:
                raise ValueError("state must be a statevector of length 2**n")
        else:
            state = torch.zeros(dim, dtype=self.dtype, device=self.device)
            state[0] = 1.0

        self.rho = state.unsqueeze(-1) @ state.conj().unsqueeze(0)
        return self

    # ------------------------------------------------------------------
    # Gate application helpers
    def _local_unitary(self, gate: torch.Tensor, qubits: Sequence[int]) -> torch.Tensor:
        gate = gate.to(device=self.device, dtype=self.dtype)
        if not qubits:
            return gate
        targets = tuple(qubits)
        cache = self._perm_cache.get(targets)
        if cache is None:
            cache = _get_perm_cache(targets, self.n_qubits)
            self._perm_cache[targets] = cache
        self.rho = _apply_kraus_density(self.rho, gate, cache)
        return gate

    def _extract_single_qubit(self, gate: torch.Tensor, q: int) -> torch.Tensor:
        if gate.shape[-1] == 2:
            return gate
        n = int(math.log2(gate.shape[-1]))
        if 1 << n != gate.shape[-1]:
            raise ValueError("Gate dimension is not a power of two")
        idx0 = 0
        idx1 = 1 << (n - q - 1)
        return gate[[idx0, idx1]][:, [idx0, idx1]]

    def _extract_two_qubit(self, gate: torch.Tensor, q1: int, q2: int) -> torch.Tensor:
        if gate.shape[-1] == 4:
            return gate
        n = int(math.log2(gate.shape[-1]))
        if 1 << n != gate.shape[-1]:
            raise ValueError("Gate dimension is not a power of two")

        def idx(b1: int, b2: int) -> int:
            return (b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1))

        sel = [idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)]
        return gate[sel][:, sel]

    # ------------------------------------------------------------------
    # QuantumBackend API
    def apply_1q(self, gate: torch.Tensor, q: int):
        gate = self._extract_single_qubit(gate, q)
        self._local_unitary(gate, [q])

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        gate = self._extract_two_qubit(gate, q1, q2)
        self._local_unitary(gate, [q1, q2])

    def apply_dense(self, gate: torch.Tensor, qubits: Sequence[int]):
        if gate.shape[-1] != 1 << len(qubits):
            raise ValueError("Gate dimension mismatch for provided qubits")
        self._local_unitary(gate, qubits)

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor:
        probs = self.rho.diagonal().real
        if qubits is None:
            return probs
        qubits = list(qubits)
        n = self.n_qubits
        out = torch.zeros(1 << len(qubits), dtype=probs.dtype, device=probs.device)
        for i, p in enumerate(probs):
            key = 0
            for j, q in enumerate(qubits):
                bit = (i >> (n - q - 1)) & 1
                key |= bit << j
            out[key] += p
        return out


__all__ = ["DensityMatrixBackend"]

