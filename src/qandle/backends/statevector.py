import math
from typing import Sequence

import torch

from .. import utils

from . import QuantumBackend
from ..kernels import apply_one_qubit, apply_two_qubit

class StateVectorBackend(QuantumBackend):
    """State-vector simulator."""

    def __init__(self, n_qubits: int, dtype=torch.complex64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.allocate(n_qubits)

    def allocate(self, n_qubits: int):
        self.n_qubits = n_qubits
        basis = torch.nn.functional.one_hot(
            torch.tensor(0, device=self.device, dtype=torch.long),
            num_classes=2 ** n_qubits,
        ).to(self.dtype)
        self.state = basis
        return self

    def _apply_gate_dense(self, gate: torch.Tensor, qubits: Sequence[int]):
        gate = gate.to(self.state)
        n = self.n_qubits
        q_be = [n - q - 1 for q in qubits]
        perm = q_be + [i for i in range(n) if i not in q_be]
        inv_perm = [perm.index(i) for i in range(n)]
        psi = self.state.view([2] * n).permute(perm).reshape(2 ** len(qubits), -1)
        psi = torch.matmul(gate, psi)
        psi = psi.reshape([2] * len(qubits) + [2] * (n - len(qubits)))
        psi = psi.permute(inv_perm).reshape(2 ** n)
        self.state = psi

    def apply_1q(self, gate: torch.Tensor, q: int):
        if gate.shape[0] != 2:
            n = int(math.log2(gate.shape[0]))
            idx0 = 0
            idx1 = 1 << (n - q - 1)
            gate = gate[[idx0, idx1]][:, [idx0, idx1]]
        self.state = apply_one_qubit(self.state, gate, q, self.n_qubits)

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        if gate.shape[0] != 4:
            n = int(math.log2(gate.shape[0]))
            idx = lambda b1, b2: ((b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1)))
            sel = [idx(0,0), idx(0,1), idx(1,0), idx(1,1)]
            gate = gate[sel][:, sel]
        self.state = apply_two_qubit(self.state, gate, q1, q2, self.n_qubits)

    def apply_dense(self, gate: torch.Tensor, qubits: Sequence[int]):
        self._apply_gate_dense(gate, qubits)

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor:
        probs = (self.state.abs() ** 2).real
        return utils.marginal_probabilities(probs, qubits, self.n_qubits)
