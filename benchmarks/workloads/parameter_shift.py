from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from qandle import operators
from qandle.autodiff import adjoint_loss_and_grad
from qandle.gradients import parameter_shift_forward
from qandle.qcircuit import Circuit
from torch.nn.utils import parameters_to_vector

_SHIFT = math.pi / 2
_COEFF = 0.5


class _ZExpectationModule(torch.nn.Module):
    """Circuit wrapper returning a single Z expectation value."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        rotations_per_qubit: int,
        *,
        split_max_qubits: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        layers: list[torch.nn.Module] = []
        rotation_cycle: Sequence[type[operators.UnbuiltParametrizedOperator]] = (
            operators.RY,
            operators.RZ,
            operators.RX,
        )
        operations: list[dict[str, object]] = []
        param_index = 0
        for layer_idx in range(depth):
            for qubit in range(num_qubits):
                for rot_idx in range(rotations_per_qubit):
                    gate_cls = rotation_cycle[(layer_idx * rotations_per_qubit + rot_idx) % len(rotation_cycle)]
                    theta = torch.rand(1, generator=rng)
                    layers.append(gate_cls(qubit, theta=theta))
                    operations.append(
                        {
                            "gate": gate_cls.__name__,
                            "qubits": [qubit],
                            "param_index": param_index,
                        }
                    )
                    param_index += int(theta.numel())
            if num_qubits > 1:
                for qubit in range(num_qubits - 1):
                    layers.append(operators.CNOT(qubit, qubit + 1))
                    operations.append({"gate": "CNOT", "qubits": [qubit, qubit + 1]})
        self.circuit = Circuit(
            layers,
            num_qubits=num_qubits,
            split_max_qubits=split_max_qubits,
        )
        self.num_qubits = num_qubits
        self.operations = operations

    def forward(self, state: torch.Tensor | None = None) -> torch.Tensor:
        psi = self.circuit(state)
        probs = psi.abs().pow(2)
        indices = torch.arange(probs.shape[-1], device=probs.device)
        bit = (indices >> (self.num_qubits - 1)) & 1
        z_eigs = (1 - 2 * bit).to(probs.dtype)
        return (probs * z_eigs).sum().to(dtype=torch.float32)


def _slow_parameter_shift(module: torch.nn.Module) -> None:
    params = [p for p in module.parameters() if p.requires_grad]
    with torch.no_grad():
        for param in params:
            grad = torch.zeros_like(param)
            flat_grad = grad.view(-1)
            flat_param = param.view(-1)
            for idx in range(flat_grad.numel()):
                value = flat_param[idx].item()
                flat_param[idx] = value + _SHIFT
                plus = module()
                flat_param[idx] = value - _SHIFT
                minus = module()
                flat_param[idx] = value
                flat_grad[idx] = _COEFF * (plus - minus)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BenchmarkResult:
    style: str
    parameters: int
    vectorized_ms: float
    naive_ms: float
    adjoint_ms: float

    @property
    def vectorized_speedup(self) -> float:
        return float("inf") if self.vectorized_ms == 0 else self.naive_ms / self.vectorized_ms

    @property
    def adjoint_speedup(self) -> float:
        return float("inf") if self.adjoint_ms == 0 else self.naive_ms / self.adjoint_ms

    @property
    def adjoint_vs_vectorized(self) -> float:
        return float("inf") if self.adjoint_ms == 0 else self.vectorized_ms / self.adjoint_ms


def _time_call(fn, repeats: int, device: torch.device) -> float:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        durations.append(time.perf_counter() - start)
    return 1000.0 * sum(durations) / len(durations)


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _vectorized_runner(module: torch.nn.Module, device: torch.device) -> None:
    module.zero_grad(set_to_none=True)
    out = parameter_shift_forward(module)
    _synchronize(device)
    out.backward()
    _synchronize(device)


def _z_observable(num_qubits: int, device: torch.device) -> torch.Tensor:
    dim = 1 << num_qubits
    indices = torch.arange(dim, device=device)
    bit = (indices >> (num_qubits - 1)) & 1
    eigs = (1 - 2 * bit).to(torch.float32)
    return torch.diag(eigs.to(dtype=torch.complex64))


def _adjoint_inputs(module: _ZExpectationModule, device: torch.device) -> tuple[torch.Tensor, dict, torch.Tensor]:
    params = parameters_to_vector([p for p in module.parameters() if p.requires_grad]).detach()
    params = params.to(device=device)
    operations: list[dict[str, object]] = []
    for entry in module.operations:
        op = {"gate": entry["gate"], "qubits": list(entry["qubits"])}
        param_index = entry.get("param_index")
        if param_index is not None:
            op["param_index"] = int(param_index)
        operations.append(op)
    description = {"n_qubits": module.num_qubits, "operations": operations}
    observable = _z_observable(module.num_qubits, device)
    return params, description, observable


def _adjoint_runner(module: _ZExpectationModule, device: torch.device) -> Callable[[], None]:
    params, description, observable = _adjoint_inputs(module, device)

    def _run() -> None:
        adjoint_loss_and_grad(params, description, observable)

    return _run


def _benchmark_style(
    *,
    parameters: int,
    num_qubits: int,
    rotations_per_qubit: int,
    style: str,
    device: torch.device,
    repeats: int,
    split_max_qubits: int,
) -> BenchmarkResult:
    stride = num_qubits * rotations_per_qubit
    if stride == 0 or parameters % stride != 0:
        raise ValueError(
            f"Cannot distribute {parameters} parameters across {num_qubits} qubits"
            f" with {rotations_per_qubit} rotations per layer."
        )
    depth = parameters // stride
    seed = 13

    vectorized_module = _ZExpectationModule(
        num_qubits=num_qubits,
        depth=depth,
        rotations_per_qubit=rotations_per_qubit,
        split_max_qubits=split_max_qubits,
        seed=seed,
    ).to(device)
    assert _count_parameters(vectorized_module) == parameters

    naive_module = _ZExpectationModule(
        num_qubits=num_qubits,
        depth=depth,
        rotations_per_qubit=rotations_per_qubit,
        split_max_qubits=split_max_qubits,
        seed=seed,
    ).to(device)

    vectorized_ms = _time_call(lambda: _vectorized_runner(vectorized_module, device), repeats, device)
    naive_ms = _time_call(lambda: _slow_parameter_shift(naive_module), max(1, repeats // 2), device)
    adjoint_ms = _time_call(_adjoint_runner(vectorized_module, device), repeats, device)

    return BenchmarkResult(
        style=style,
        parameters=parameters,
        vectorized_ms=vectorized_ms,
        naive_ms=naive_ms,
        adjoint_ms=adjoint_ms,
    )


def run_parameter_shift_benchmarks(
    device: torch.device,
    repeats: int,
    split_max_qubits: int,
) -> Iterable[BenchmarkResult]:
    num_qubits = 4
    param_counts = (8, 32, 128)
    styles = (
        ("shallow", 2),
        ("deep", 1),
    )
    for style_name, rotations in styles:
        for params in param_counts:
            yield _benchmark_style(
                parameters=params,
                num_qubits=num_qubits,
                rotations_per_qubit=rotations,
                style=style_name,
                device=device,
                repeats=repeats,
                split_max_qubits=split_max_qubits,
            )


def _format_row(result: BenchmarkResult) -> str:
    return (
        f"{result.style:<8} | {result.parameters:>4} | {result.vectorized_ms:>9.3f} | "
        f"{result.naive_ms:>9.3f} | {result.adjoint_ms:>9.3f} | "
        f"{result.vectorized_speedup:>7.2f}x | {result.adjoint_speedup:>7.2f}x | "
        f"{result.adjoint_vs_vectorized:>7.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parameter-shift (vectorized and scalar) and adjoint gradients."
    )
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per measurement")
    parser.add_argument(
        "--split-max-qubits",
        type=int,
        default=0,
        help="Optional subcircuit splitting threshold to exercise split circuits.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but not available.")

    print("style    |    P | vec_ms   | naive_ms | adj_ms   | n/vec  | n/adj  | vec/adj")
    print("-" * 82)
    for result in run_parameter_shift_benchmarks(device, args.repeats, args.split_max_qubits):
        print(_format_row(result))


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
