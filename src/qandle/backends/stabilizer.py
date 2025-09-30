import math
from typing import List, Sequence, Tuple

import torch

from . import QuantumBackend
from .statevector import StateVectorBackend

_PHASES: List[complex] = [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j]
_PAULI_LABELS = {0: "I", 1: "X", 2: "Y", 3: "Z"}


def _phase_index(value: complex, atol: float = 1e-6) -> int:
    for idx, phase in enumerate(_PHASES):
        if abs(value - phase) <= atol:
            return idx
    raise ValueError(f"Phase {value} not close to Clifford phase multiples.")


def _remove_global_phase(gate: torch.Tensor) -> torch.Tensor:
    gate = gate.clone().to(torch.complex64)
    non_zero = torch.nonzero(gate.abs() > 1e-8, as_tuple=False)
    if non_zero.numel() == 0:
        return gate
    i0, i1 = non_zero[0].tolist()
    ref = gate[i0, i1]
    magnitude = ref.abs()
    if magnitude <= 1e-8:
        return gate
    return gate / (ref / magnitude)


_PAULI_MATRICES = {
    0: torch.eye(2, dtype=torch.complex64),
    1: torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    2: torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    3: torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
}

_TWO_Q_PAULIS = {
    (p, q): torch.kron(_PAULI_MATRICES[p], _PAULI_MATRICES[q]) for p in range(4) for q in range(4)
}


def _conjugate_1q_pauli(pauli: int, gate: torch.Tensor, atol: float = 1e-6) -> Tuple[int, int]:
    base = _PAULI_MATRICES[pauli]
    conj = gate @ base @ gate.conj().transpose(0, 1)
    for new_id, target in _PAULI_MATRICES.items():
        inner = torch.trace(target.conj().transpose(0, 1) @ conj) / 2
        value = complex(inner.item())
        if abs(value) <= atol:
            continue
        try:
            phase_idx = _phase_index(value)
        except ValueError:
            continue
        if torch.allclose(conj, target * _PHASES[phase_idx], atol=atol):
            return new_id, phase_idx
    raise ValueError("Gate is not Clifford on the given Pauli basis element.")


def _conjugate_2q_pauli(p1: int, p2: int, gate: torch.Tensor, atol: float = 1e-6) -> Tuple[int, int, int]:
    base = _TWO_Q_PAULIS[(p1, p2)]
    conj = gate @ base @ gate.conj().transpose(0, 1)
    for (n1, n2), target in _TWO_Q_PAULIS.items():
        inner = torch.trace(target.conj().transpose(0, 1) @ conj) / 4
        value = complex(inner.item())
        if abs(value) <= atol:
            continue
        try:
            phase_idx = _phase_index(value)
        except ValueError:
            continue
        if torch.allclose(conj, target * _PHASES[phase_idx], atol=atol):
            return n1, n2, phase_idx
    raise ValueError("Gate is not Clifford on the given Pauli basis element.")


def _extract_1q_gate(gate: torch.Tensor, q: int) -> torch.Tensor:
    if gate.shape[0] == 2:
        return gate.to(torch.complex64)
    n = int(round(math.log2(gate.shape[0])))
    idx0 = 0
    idx1 = 1 << (n - q - 1)
    sub = gate[[idx0, idx1]][:, [idx0, idx1]]
    return sub.to(torch.complex64)


def _extract_2q_gate(gate: torch.Tensor, q1: int, q2: int) -> torch.Tensor:
    if gate.shape[0] == 4:
        return gate.to(torch.complex64)
    n = int(round(math.log2(gate.shape[0])))
    sel = []
    for b1 in (0, 1):
        for b2 in (0, 1):
            idx = (b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1))
            sel.append(idx)
    sub = gate[sel][:, sel]
    return sub.to(torch.complex64)


class StabilizerBackend(QuantumBackend):
    """Backend that simulates Clifford circuits using a stabilizer tableau."""

    requires_clifford = True

    def __init__(self, n_qubits: int, *, dtype: torch.dtype = torch.complex64, device: str | torch.device = "cpu"):
        self.dtype = dtype
        self.device = device
        self.allocate(n_qubits)

    def allocate(self, n_qubits: int):
        self.n_qubits = n_qubits
        self._rows = torch.zeros((2 * n_qubits, n_qubits), dtype=torch.int8)
        for i in range(n_qubits):
            self._rows[i, i] = 1  # Destabilizers (X_i)
            self._rows[n_qubits + i, i] = 3  # Stabilizers (Z_i)
        self._phases = torch.zeros(2 * n_qubits, dtype=torch.int8)
        self._history: List[tuple] = []
        return self

    # Helper utilities -------------------------------------------------
    def tableau(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a copy of the tableau rows and phase exponents."""
        return self._rows.clone(), self._phases.clone()

    def pauli_string(self, row: int) -> str:
        return "".join(_PAULI_LABELS[int(p)] for p in self._rows[row])

    # Validation -------------------------------------------------------
    @staticmethod
    def is_clifford_1q(gate: torch.Tensor) -> bool:
        gate = _remove_global_phase(gate)
        try:
            for pauli in (1, 2, 3):
                _conjugate_1q_pauli(pauli, gate)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_clifford_2q(gate: torch.Tensor) -> bool:
        gate = _remove_global_phase(gate)
        try:
            for paulis in ((1, 0), (0, 1), (3, 0), (0, 3)):
                _conjugate_2q_pauli(paulis[0], paulis[1], gate)
            return True
        except ValueError:
            return False

    def _prepare_1q_gate(self, gate: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = _extract_1q_gate(gate, q)
        canon = _remove_global_phase(sub)
        if not self.is_clifford_1q(canon):
            raise ValueError("Only single-qubit Clifford gates are supported by the stabilizer backend.")
        return canon, sub.to(dtype=self.dtype, device=self.device)

    def _prepare_2q_gate(self, gate: torch.Tensor, q1: int, q2: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = _extract_2q_gate(gate, q1, q2)
        canon = _remove_global_phase(sub)
        if not self.is_clifford_2q(canon):
            raise ValueError("Only two-qubit Clifford gates are supported by the stabilizer backend.")
        return canon, sub.to(dtype=self.dtype, device=self.device)

    def validate_1q_gate(self, gate: torch.Tensor, q: int) -> None:
        self._prepare_1q_gate(gate, q)

    def validate_2q_gate(self, gate: torch.Tensor, q1: int, q2: int) -> None:
        self._prepare_2q_gate(gate, q1, q2)

    # Gate application -------------------------------------------------
    def apply_1q(self, gate: torch.Tensor, q: int):
        canon_gate, raw_gate = self._prepare_1q_gate(gate, q)
        for row in range(self._rows.shape[0]):
            pauli = int(self._rows[row, q])
            new_pauli, phase_idx = _conjugate_1q_pauli(pauli, canon_gate)
            self._rows[row, q] = new_pauli
            self._phases[row] = (self._phases[row] + phase_idx) % 4
        self._history.append(("1q", q, raw_gate))

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        canon_gate, raw_gate = self._prepare_2q_gate(gate, q1, q2)
        for row in range(self._rows.shape[0]):
            pauli1 = int(self._rows[row, q1])
            pauli2 = int(self._rows[row, q2])
            new1, new2, phase_idx = _conjugate_2q_pauli(pauli1, pauli2, canon_gate)
            self._rows[row, q1] = new1
            self._rows[row, q2] = new2
            self._phases[row] = (self._phases[row] + phase_idx) % 4
        self._history.append(("2q", q1, q2, raw_gate))

    # Measurement ------------------------------------------------------
    def _simulate_statevector(self) -> StateVectorBackend:
        backend = StateVectorBackend(self.n_qubits, dtype=self.dtype, device=self.device)
        for op in self._history:
            if op[0] == "1q":
                backend.apply_1q(op[2], op[1])
            else:
                backend.apply_2q(op[3], op[1], op[2])
        return backend

    def to_statevector(self) -> torch.Tensor:
        return self._simulate_statevector().state

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor:
        backend = self._simulate_statevector()
        return backend.measure(qubits)
