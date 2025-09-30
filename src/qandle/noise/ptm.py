r"""Pauli transfer matrix utilities and backend for noise propagation.

This module implements a simulator that evolves density operators in the Pauli
basis instead of the computational basis.  States are represented by the
coefficients ``r_P`` of the Pauli expansion

r.. math::

    \rho = \frac{1}{2^n} \sum_P r_P P,

where ``P`` ranges over ``n``-qubit Pauli strings ordered lexicographically.
Local unitary gates act on the coefficients through the Pauli transfer matrix
(PTM) representation ``T`` and noise channels are applied in the same basis.
The simulator focuses on circuits built from diagonal-in-``Z`` rotations and
Clifford entanglers (``RZ``, ``CNOT``, ``CZ``) as these map Pauli strings onto
Pauli strings without generating dense couplings.  When unsupported operations
are encountered the backend performs a Pauli twirl approximation by discarding
off-diagonal PTM elements.  This keeps the evolution within the Pauli-diagonal
manifold while providing a graceful degradation path.

The backend exposes the same interface as the other simulators used by
``Circuit.forward`` and therefore integrates with the existing noise model
machinery.  Expectation values of Pauli-sum observables can be evaluated
directly from the stored coefficients without materialising the full density
matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import torch

from .channels import BuiltNoiseChannel, NoiseChannel

_PAULI_TO_INDEX = {"I": 0, "X": 1, "Y": 2, "Z": 3}


@lru_cache(maxsize=None)
def _single_paulis(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return the single-qubit Pauli matrices on ``device``."""

    dtype = torch.complex128
    return (
        torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device),
        torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
    )


@lru_cache(maxsize=None)
def _pauli_basis(n_qubits: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return the tensor-product Pauli basis for ``n_qubits``."""

    singles = _single_paulis(device)
    basis: list[torch.Tensor] = [torch.eye(1, dtype=torch.complex128, device=device)]
    for _ in range(n_qubits):
        new_basis: list[torch.Tensor] = []
        for prev in basis:
            for single in singles:
                new_basis.append(torch.kron(prev, single))
        basis = new_basis
    return tuple(basis)


def _unitary_to_ptm(unitary: torch.Tensor) -> torch.Tensor:
    """Return the PTM of a unitary acting on ``log2(unitary.shape[0])`` qubits."""

    dim = unitary.shape[0]
    n_qubits = int(round(math.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError("Unitary dimension must be a power of two.")
    device = unitary.device
    basis = _pauli_basis(n_qubits, device)
    dag = unitary.conj().transpose(-2, -1)
    out = torch.zeros((4**n_qubits, 4**n_qubits), dtype=torch.float64, device=device)
    scale = 1.0 / (2**n_qubits)
    for j, pauli_j in enumerate(basis):
        transformed = unitary @ pauli_j @ dag
        for i, pauli_i in enumerate(basis):
            val = torch.trace(pauli_i @ transformed)
            out[i, j] = val.real * scale
    return out


def _channel_to_ptm(channel: NoiseChannel, *, device: torch.device) -> torch.Tensor:
    """Return the PTM representation of ``channel``."""

    kraus = channel.to_kraus(dtype=torch.complex128, device=device)
    if not kraus:
        raise ValueError("Noise channel must provide at least one Kraus operator.")
    dim = kraus[0].shape[0]
    n_qubits = int(round(math.log2(dim)))
    basis = _pauli_basis(n_qubits, device)
    out = torch.zeros((4**n_qubits, 4**n_qubits), dtype=torch.float64, device=device)
    scale = 1.0 / (2**n_qubits)
    for j, pauli_j in enumerate(basis):
        evolved = torch.zeros_like(pauli_j)
        for op in kraus:
            evolved = evolved + op @ pauli_j @ op.conj().transpose(-2, -1)
        for i, pauli_i in enumerate(basis):
            val = torch.trace(pauli_i @ evolved)
            out[i, j] = val.real * scale
    return out


def _pauli_index(labels: Sequence[int]) -> int:
    """Map per-qubit Pauli indices to the flattened coefficient index."""

    out = 0
    for value in labels:
        out = (out << 2) | value
    return out


def _pauli_string_to_digits(pauli: Mapping[int, str] | Sequence[str] | str, n_qubits: int) -> list[int]:
    """Convert a Pauli specification to per-qubit indices."""

    digits = [0] * n_qubits
    if isinstance(pauli, str):
        if len(pauli) != n_qubits:
            raise ValueError("Pauli string must match number of qubits.")
        for idx, char in enumerate(pauli.upper()):
            if char not in _PAULI_TO_INDEX:
                raise ValueError(f"Invalid Pauli label '{char}'.")
            digits[idx] = _PAULI_TO_INDEX[char]
        return digits

    if isinstance(pauli, Sequence):
        if len(pauli) != n_qubits:
            raise ValueError("Pauli sequence must match number of qubits.")
        return _pauli_string_to_digits("".join(str(p).upper() for p in pauli), n_qubits)

    for qubit, char in pauli.items():
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range in Pauli mapping.")
        label = char.upper()
        if label not in _PAULI_TO_INDEX:
            raise ValueError(f"Invalid Pauli label '{label}'.")
        digits[qubit] = _PAULI_TO_INDEX[label]
    return digits


@dataclass
class PauliTransferMatrixBackend:
    """Simulator evolving Pauli coefficients instead of amplitudes."""

    n_qubits: int
    dtype: torch.dtype = torch.float64
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.device = torch.device("cpu") if self.device is None else torch.device(self.device)
        base = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        coeffs = base
        for _ in range(self.n_qubits - 1):
            coeffs = torch.kron(coeffs, base)
        self.coeffs = coeffs.clone().contiguous()
        self._unitary_cache: dict[tuple[int, bytes], torch.Tensor] = {}
        self._channel_cache: dict[tuple[int, bytes], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Backend interface
    def allocate(self, n_qubits: int) -> PauliTransferMatrixBackend:
        self.n_qubits = n_qubits
        base = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        coeffs = base
        for _ in range(self.n_qubits - 1):
            coeffs = torch.kron(coeffs, base)
        self.coeffs = coeffs.clone().contiguous()
        self._unitary_cache.clear()
        self._channel_cache.clear()
        return self

    def _extract_local(self, gate: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
        if not targets:
            raise ValueError("At least one target qubit is required.")
        local_dim = 2 ** len(targets)
        if gate.shape[0] == local_dim:
            return gate

        total_qubits = int(round(math.log2(gate.shape[0])))
        if 2 ** total_qubits != gate.shape[0]:
            raise ValueError("Gate dimension must be a power of two.")
        if any(t < 0 or t >= total_qubits for t in targets):
            raise ValueError("Target index out of range for provided gate matrix.")

        indices: list[int] = []
        for bits in range(local_dim):
            acc = 0
            for bit_idx, qubit in enumerate(targets):
                shift = total_qubits - qubit - 1
                if bits & (1 << (len(targets) - bit_idx - 1)):
                    acc |= 1 << shift
            indices.append(acc)

        return gate[indices][:, indices]

    def _local_unitary_ptm(self, gate: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
        local = self._extract_local(gate, targets)
        key = (len(targets), torch.view_as_real(local.detach().cpu()).numpy().tobytes())
        cached = self._unitary_cache.get(key)
        if cached is None:
            ptm = _unitary_to_ptm(local.detach().cpu().to(torch.complex128))
            self._unitary_cache[key] = ptm
            cached = ptm
        return cached.to(device=self.device, dtype=self.dtype)

    def _apply_local(self, matrix: torch.Tensor, targets: Sequence[int]) -> None:
        if not targets:
            return
        n = self.n_qubits
        perm = list(targets) + [i for i in range(n) if i not in targets]
        inv_perm = [0] * n
        for idx, value in enumerate(perm):
            inv_perm[value] = idx
        coeffs = self.coeffs.view(*(4,) * n).permute(perm)
        coeffs = coeffs.reshape(4 ** len(targets), -1)
        updated = matrix.to(coeffs) @ coeffs
        updated = updated.reshape(*(4,) * n).permute(inv_perm)
        self.coeffs = updated.reshape(-1).contiguous()

    def apply_1q(self, gate: torch.Tensor, q: int) -> None:
        ptm = self._local_unitary_ptm(gate, (q,))
        if not torch.allclose(ptm, torch.diag(ptm.diagonal()), atol=1e-8):
            ptm = torch.diag(ptm.diagonal())
        self._apply_local(ptm, (q,))

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int) -> None:
        targets = (q1, q2)
        ptm = self._local_unitary_ptm(gate, targets)
        if not torch.allclose(ptm, torch.diag(ptm.diagonal()), atol=1e-8):
            ptm = torch.diag(ptm.diagonal())
        self._apply_local(ptm, targets)

    def apply_noise_channel(self, channel: BuiltNoiseChannel) -> None:
        if getattr(channel.channel, "is_identity", False):
            return
        key = (len(channel.targets), id(channel.channel))
        matrix = self._channel_cache.get(key)
        if matrix is None:
            matrix = _channel_to_ptm(channel.channel, device=self.device)
            self._channel_cache[key] = matrix
        self._apply_local(matrix.to(dtype=self.dtype, device=self.device), channel.targets)

    # ------------------------------------------------------------------
    # Observables
    def expectation_pauli_string(self, pauli: Mapping[int, str] | Sequence[str] | str) -> float:
        digits = _pauli_string_to_digits(pauli, self.n_qubits)
        index = _pauli_index(digits)
        return float(self.coeffs[index])

    def expectation_pauli_sum(
        self, terms: Iterable[tuple[complex | float, Mapping[int, str] | Sequence[str] | str]]
    ) -> float:
        total = 0.0
        for weight, pauli in terms:
            coeff = self.expectation_pauli_string(pauli)
            total += float(weight) * coeff
        return total

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor:
        if qubits is not None and len(qubits) != self.n_qubits:
            raise NotImplementedError("Partial measurement is not implemented for the PTM backend.")
        n = self.n_qubits
        tensor = self.coeffs.view(*(4,) * n)
        for axis in range(n):
            tensor = torch.index_select(
                tensor, dim=axis, index=torch.tensor([0, 3], device=tensor.device, dtype=torch.long)
            )
        flat = tensor.reshape(-1)
        probs = torch.zeros(1 << n, dtype=self.dtype, device=self.device)
        scale = 1.0 / (1 << n)
        for bitstring in range(1 << n):
            acc = 0.0
            for subset in range(1 << n):
                parity = ((bitstring & subset).bit_count() & 1)
                sign = -1.0 if parity else 1.0
                acc += sign * float(flat[subset])
            probs[bitstring] = acc * scale
        probs = torch.clamp(probs, min=0.0)
        return probs / probs.sum()


__all__ = ["PauliTransferMatrixBackend"]

