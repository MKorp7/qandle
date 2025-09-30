"""Adjoint differentiation for variational quantum circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

from ..kernels import apply_one_qubit, apply_two_qubit


_PAULI_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
_PAULI_Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64)
_PAULI_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
_TWO_QUBIT_ID = torch.eye(4, dtype=torch.complex64)
_ONE_QUBIT_ID = torch.eye(2, dtype=torch.complex64)


@dataclass(frozen=True)
class GateOperation:
    """Description of a circuit operation."""

    gate: str
    qubits: Sequence[int]
    param_index: Optional[int] = None


RotationGenerators = {
    "RX": _PAULI_X,
    "RY": _PAULI_Y,
    "RZ": _PAULI_Z,
    "ZZ": torch.kron(_PAULI_Z, _PAULI_Z),
}


_FIXED_GATES = {
    "CNOT": torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.complex64,
    ),
    "CZ": torch.diag(torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.complex64)),
}


def _identity(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size == 2:
        return _ONE_QUBIT_ID.to(device=device, dtype=dtype)
    if size == 4:
        return _TWO_QUBIT_ID.to(device=device, dtype=dtype)
    return torch.eye(size, dtype=dtype, device=device)


def _rotation_unitary(generator: torch.Tensor, theta: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = generator.to(device=device, dtype=dtype)
    identity = _identity(generator.shape[0], device=device, dtype=dtype)
    theta = theta.to(dtype=torch.float32, device=device)
    half_theta = theta / 2
    cos = torch.cos(half_theta).to(dtype)
    sin = torch.sin(half_theta).to(dtype)
    return cos * identity - 1j * sin * generator


def _gate_unitary(gate: str, theta: Optional[torch.Tensor], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if gate in RotationGenerators:
        if theta is None:
            raise ValueError(f"Gate '{gate}' expects an associated parameter")
        return _rotation_unitary(RotationGenerators[gate], theta, device=device, dtype=dtype)
    if gate in _FIXED_GATES:
        return _FIXED_GATES[gate].to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported gate '{gate}'")


def _gate_derivative(
    gate: str,
    theta: Optional[torch.Tensor],
    unitary: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if gate not in RotationGenerators:
        return None
    if theta is None:
        raise ValueError(f"Gate '{gate}' expects an associated parameter")
    generator = RotationGenerators[gate].to(device=device, dtype=dtype)
    return (-0.5j) * generator @ unitary


def _apply_gate(state: torch.Tensor, gate: torch.Tensor, qubits: Sequence[int], n_qubits: int) -> torch.Tensor:
    gate = gate.to(device=state.device, dtype=state.dtype)
    if len(qubits) == 1:
        return apply_one_qubit(state, gate, qubits[0], n_qubits)
    if len(qubits) == 2:
        return apply_two_qubit(state, gate, qubits[0], qubits[1], n_qubits)
    raise ValueError("Only one- and two-qubit gates are supported")


def _apply_gate_dagger(state: torch.Tensor, gate: torch.Tensor, qubits: Sequence[int], n_qubits: int) -> torch.Tensor:
    return _apply_gate(state, gate.conj().transpose(-2, -1), qubits, n_qubits)


def _initial_state(n_qubits: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    state = torch.zeros(2**n_qubits, dtype=dtype, device=device)
    state[0] = 1.0
    return state


class AdjointFunction(torch.autograd.Function):
    """Custom autograd function implementing the adjoint differentiation rule."""

    @staticmethod
    def forward(
        ctx,
        params: torch.Tensor,
        circuit_description: dict,
        init_state: Optional[torch.Tensor],
        observable: torch.Tensor,
        checkpoint_stride: int = 1,
    ) -> torch.Tensor:
        del checkpoint_stride  # currently unused: full state storage is employed.
        n_qubits = int(circuit_description["n_qubits"])
        ops: Iterable[GateOperation | dict] = circuit_description["operations"]
        device = params.device
        dtype = observable.dtype if observable.is_complex() else torch.complex64

        with torch.no_grad():
            state = (
                _initial_state(n_qubits, device=device, dtype=dtype)
                if init_state is None
                else init_state.to(device=device, dtype=dtype)
            )
            states: List[torch.Tensor] = [state.clone()]
            unitaries: List[torch.Tensor] = []
            prepared_ops: List[GateOperation] = []

            for entry in ops:
                if isinstance(entry, GateOperation):
                    op = entry
                else:
                    op = GateOperation(
                        gate=entry["gate"],
                        qubits=tuple(entry["qubits"]),
                        param_index=entry.get("param_index"),
                    )
                theta = None
                if op.param_index is not None:
                    theta = params[op.param_index]
                unitary = _gate_unitary(op.gate, theta, device=device, dtype=dtype)
                state = _apply_gate(state, unitary, op.qubits, n_qubits)
                states.append(state.clone())
                unitaries.append(unitary.clone())
                prepared_ops.append(op)

            observable = observable.to(device=device, dtype=dtype)
            final_state = states[-1]
            obs_psi = observable @ final_state
            loss = torch.vdot(final_state, obs_psi).real.to(params.dtype)

        ctx.save_for_backward(params.detach())
        ctx.states = states
        ctx.unitaries = unitaries
        ctx.ops = prepared_ops
        ctx.observable = observable
        ctx.n_qubits = n_qubits
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (params_detached,) = ctx.saved_tensors
        states: List[torch.Tensor] = ctx.states
        unitaries: List[torch.Tensor] = ctx.unitaries
        ops: List[GateOperation] = ctx.ops
        observable: torch.Tensor = ctx.observable
        n_qubits: int = ctx.n_qubits

        device = states[0].device
        dtype = states[0].dtype
        lam = observable @ states[-1]
        grads = torch.zeros_like(params_detached, dtype=params_detached.dtype)

        for idx in reversed(range(len(ops))):
            op = ops[idx]
            unitary = unitaries[idx]
            psi_prev = states[idx]

            if op.param_index is not None:
                theta = params_detached[op.param_index]
                dU = _gate_derivative(op.gate, theta, unitary, device=device, dtype=dtype)
                if dU is not None:
                    dpsi = _apply_gate(psi_prev, dU, op.qubits, n_qubits)
                    contrib = (2.0 * torch.vdot(lam, dpsi).real).to(grads.dtype)
                    grads[op.param_index] = grads[op.param_index] + contrib

            lam = _apply_gate_dagger(lam, unitary, op.qubits, n_qubits)

        grads = grad_output.real * grads
        return grads, None, None, None, None


def adjoint_expectation(
    params: torch.Tensor,
    circuit_description: dict,
    observable: torch.Tensor,
    *,
    init_state: Optional[torch.Tensor] = None,
    checkpoint_stride: int = 1,
) -> torch.Tensor:
    """Return the expectation value using the adjoint simulator."""

    return AdjointFunction.apply(params, circuit_description, init_state, observable, checkpoint_stride)


def adjoint_loss_and_grad(
    params: torch.Tensor,
    circuit_description: dict,
    observable: torch.Tensor,
    *,
    init_state: Optional[torch.Tensor] = None,
    checkpoint_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience helper returning both the loss and its gradient."""

    params = params.clone().detach().requires_grad_(True)
    loss = AdjointFunction.apply(params, circuit_description, init_state, observable, checkpoint_stride)
    loss.backward()
    grad = params.grad.detach().clone()
    return loss.detach(), grad


class AdjointCircuitModule(torch.nn.Module):
    """Thin :class:`~torch.nn.Module` wrapper around :class:`AdjointFunction`."""

    def __init__(
        self,
        circuit_description: dict,
        observable: torch.Tensor,
        *,
        init_state: Optional[torch.Tensor] = None,
        checkpoint_stride: int = 1,
    ) -> None:
        super().__init__()
        self.circuit_description = circuit_description
        self.register_buffer("observable", observable)
        if init_state is not None:
            self.register_buffer("init_state", init_state)
        else:
            self.init_state = None  # type: ignore[assignment]
        self.checkpoint_stride = checkpoint_stride

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return AdjointFunction.apply(
            params,
            self.circuit_description,
            getattr(self, "init_state", None),
            self.observable,
            self.checkpoint_stride,
        )


__all__ = [
    "AdjointFunction",
    "AdjointCircuitModule",
    "adjoint_expectation",
    "adjoint_loss_and_grad",
]
