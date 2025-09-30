import math
import time
from typing import Dict, List, Tuple

import torch

from qandle.autodiff import adjoint_expectation
from qandle.autodiff.adjoint import AdjointFunction


def _random_observable(n_qubits: int, generator: torch.Generator) -> torch.Tensor:
    dim = 1 << n_qubits
    real = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    imag = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
    mat = real + 1j * imag
    hermitian = (mat + mat.conj().transpose(-2, -1)) / 2
    return hermitian.to(dtype=torch.complex64)


def _random_circuit_description(n_qubits: int, depth: int, generator: torch.Generator) -> Tuple[Dict, int]:
    operations: List[Dict] = []
    param_index = 0
    single_qubit_gates = ["RX", "RY", "RZ"]
    for _ in range(depth):
        for qubit in range(n_qubits):
            gate = single_qubit_gates[torch.randint(0, len(single_qubit_gates), (1,), generator=generator).item()]
            operations.append({"gate": gate, "qubits": [qubit], "param_index": param_index})
            param_index += 1
        for qubit in range(n_qubits - 1):
            operations.append({"gate": "CNOT", "qubits": [qubit, qubit + 1]})
    circuit = {"n_qubits": n_qubits, "operations": operations}
    return circuit, param_index


def _hardware_efficient_ansatz(n_qubits: int, layers: int) -> Tuple[Dict, int]:
    operations: List[Dict] = []
    param_index = 0
    for _ in range(layers):
        for qubit in range(n_qubits):
            operations.append({"gate": "RY", "qubits": [qubit], "param_index": param_index})
            param_index += 1
        for qubit in range(n_qubits):
            operations.append({"gate": "RZ", "qubits": [qubit], "param_index": param_index})
            param_index += 1
        for qubit in range(n_qubits - 1):
            operations.append({"gate": "CNOT", "qubits": [qubit, qubit + 1]})
    return {"n_qubits": n_qubits, "operations": operations}, param_index


def _finite_difference_grad(params: torch.Tensor, circuit: Dict, observable: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    grads = torch.zeros_like(params)
    with torch.no_grad():
        for idx in range(params.numel()):
            theta_plus = params.clone()
            theta_minus = params.clone()
            theta_plus[idx] += eps
            theta_minus[idx] -= eps
            f_plus = adjoint_expectation(theta_plus, circuit, observable)
            f_minus = adjoint_expectation(theta_minus, circuit, observable)
            grads[idx] = (f_plus - f_minus) / (2 * eps)
    return grads


def _parameter_shift_grad(params: torch.Tensor, circuit: Dict, observable: torch.Tensor, shift: float = math.pi / 2) -> torch.Tensor:
    grads = torch.zeros_like(params)
    with torch.no_grad():
        for idx in range(params.numel()):
            theta_plus = params.clone()
            theta_minus = params.clone()
            theta_plus[idx] += shift
            theta_minus[idx] -= shift
            f_plus = adjoint_expectation(theta_plus, circuit, observable)
            f_minus = adjoint_expectation(theta_minus, circuit, observable)
            grads[idx] = 0.5 * (f_plus - f_minus)
    return grads


def test_adjoint_matches_finite_difference():
    generator = torch.Generator().manual_seed(1234)
    n_qubits = 4
    depth = 3
    circuit, num_params = _random_circuit_description(n_qubits, depth, generator)
    observable = _random_observable(n_qubits, generator)
    params = torch.randn(num_params, generator=generator, dtype=torch.float32, requires_grad=True)

    loss = AdjointFunction.apply(params, circuit, None, observable, 1)
    loss.backward()
    adjoint_grad = params.grad.detach().clone()

    fd_grad = _finite_difference_grad(params.detach(), circuit, observable)
    torch.testing.assert_close(adjoint_grad, fd_grad, atol=5e-3, rtol=5e-3)


def test_adjoint_matches_parameter_shift():
    generator = torch.Generator().manual_seed(2024)
    n_qubits = 6
    layers = 8
    circuit, num_params = _hardware_efficient_ansatz(n_qubits, layers)
    observable = _random_observable(n_qubits, generator)
    params = torch.randn(num_params, generator=generator, dtype=torch.float32, requires_grad=True)

    loss = AdjointFunction.apply(params, circuit, None, observable, 1)
    loss.backward()
    adjoint_grad = params.grad.detach().clone()

    shift_grad = _parameter_shift_grad(params.detach(), circuit, observable)
    torch.testing.assert_close(adjoint_grad, shift_grad, atol=5e-4, rtol=5e-4)


def test_adjoint_beats_parameter_shift_benchmark():
    generator = torch.Generator().manual_seed(31415)
    n_qubits = 6
    layers = 8
    circuit, num_params = _hardware_efficient_ansatz(n_qubits, layers)
    observable = _random_observable(n_qubits, generator)
    params = torch.randn(num_params, generator=generator, dtype=torch.float32, requires_grad=True)

    # Warm-up to stabilise timing
    for _ in range(2):
        params.grad = None
        loss = AdjointFunction.apply(params, circuit, None, observable, 1)
        loss.backward()

    params.grad = None
    start = time.perf_counter()
    loss = AdjointFunction.apply(params, circuit, None, observable, 1)
    loss.backward()
    adjoint_time = time.perf_counter() - start

    ps_start = time.perf_counter()
    _ = _parameter_shift_grad(params.detach(), circuit, observable)
    param_shift_time = time.perf_counter() - ps_start

    ratio = adjoint_time / param_shift_time
    assert ratio <= 0.4, f"Adjoint differentiation expected to be at least 2.5x faster, got ratio {ratio:.3f}"
