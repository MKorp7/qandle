import torch
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class QuantumBackend(Protocol):
    """Backend interface."""

    def allocate(self, n_qubits: int): ...

    def apply_1q(self, gate: torch.Tensor, q: int): ...

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int): ...

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor: ...


__all__ = [
    "QuantumBackend",
    "StateVectorBackend",
    "MPSBackend",
    "DensityMatrixBackend",
    "StabilizerBackend",
    "OOCStateVectorSimulator",
    "PauliTransferMatrixBackend",
]

from .statevector import StateVectorBackend
from .mps import MPSBackend
from .density_matrix import DensityMatrixBackend
from .stabilizer import StabilizerBackend
from .ooc import OOCStateVectorSimulator
from qandle.noise.ptm import PauliTransferMatrixBackend
