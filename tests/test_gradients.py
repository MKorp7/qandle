from __future__ import annotations

import math
from typing import Iterable, List

import pytest

torch = pytest.importorskip("torch")

from qandle import operators
from qandle.gradients import parameter_shift_forward
from qandle.qcircuit import Circuit


_SHIFT = math.pi / 2
_COEFF = 0.5


class _ZExpectationCircuit(torch.nn.Module):
    """Wrap a :class:`qandle.qcircuit.Circuit` and return a single Z expectation value."""

    def __init__(self, num_qubits: int, depth: int, split_max_qubits: int = 0, seed: int = 0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        layers: List[torch.nn.Module] = []
        for _ in range(depth):
            for qubit in range(num_qubits):
                theta = torch.rand(1, generator=rng)
                layers.append(operators.RY(qubit, theta=theta))
            for qubit in range(num_qubits - 1):
                layers.append(operators.CNOT(qubit, qubit + 1))
        self.circuit = Circuit(layers, num_qubits=num_qubits, split_max_qubits=split_max_qubits)
        self.num_qubits = num_qubits

    def forward(self, state: torch.Tensor | None = None) -> torch.Tensor:
        psi = self.circuit(state)
        probs = psi.abs().pow(2)
        indices = torch.arange(probs.shape[-1], device=probs.device)
        # qubit indexing in qandle uses big-endian order
        bit = (indices >> (self.num_qubits - 1)) & 1
        z_eigs = (1 - 2 * bit).to(probs.dtype)
        expval = (probs * z_eigs).sum()
        return expval.to(dtype=torch.float32)


def _slow_parameter_shift(
    module: torch.nn.Module,
    state: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    params = [p for p in module.parameters() if p.requires_grad]
    with torch.no_grad():
        for param in params:
            grad_for_p = torch.zeros_like(param)
            flat = grad_for_p.view(-1)
            orig = param.view(-1)
            for idx in range(flat.numel()):
                value = orig[idx].item()
                orig[idx] = value + _SHIFT
                plus = module(state) if state is not None else module()
                orig[idx] = value - _SHIFT
                minus = module(state) if state is not None else module()
                orig[idx] = value
                flat[idx] = _COEFF * (plus - minus)
            grads.append(grad_for_p)
    return grads


def _vectorized_grads(
    module: torch.nn.Module,
    state: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    module.zero_grad(set_to_none=True)
    out = parameter_shift_forward(module, state)
    out.backward()
    grads: List[torch.Tensor] = []
    for param in module.parameters():
        if param.requires_grad:
            grads.append(param.grad.detach().clone())
    return grads


def _assert_allclose(reference: Iterable[torch.Tensor], test: Iterable[torch.Tensor]) -> None:
    for ref, tst in zip(reference, test):
        torch.testing.assert_close(ref, tst, rtol=1e-6, atol=1e-6)


def test_parameter_shift_matches_naive():
    circuit = _ZExpectationCircuit(num_qubits=3, depth=3, seed=123)
    slow = _slow_parameter_shift(circuit)
    fast = _vectorized_grads(circuit)
    _assert_allclose(slow, fast)


def test_parameter_shift_supports_splitted_circuit():
    circuit = _ZExpectationCircuit(num_qubits=3, depth=3, split_max_qubits=1, seed=321)
    slow = _slow_parameter_shift(circuit)
    fast = _vectorized_grads(circuit)
    _assert_allclose(slow, fast)
