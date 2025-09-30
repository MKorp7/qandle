"""Microbenchmarks for dense vs. sparse control-gate application."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch

import qandle.config as config
import qandle.operators as op


@dataclass
class BenchmarkResult:
    build_time: float
    apply_time: float


def _time_it(fn, repeats: int = 5) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():  # pragma: no cover - GPU sync only when available
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


@contextlib.contextmanager
def _dense_flag(enabled: bool):
    prev = config.USE_DENSE_CONTROL_GATES
    config.USE_DENSE_CONTROL_GATES = enabled
    try:
        yield
    finally:
        config.USE_DENSE_CONTROL_GATES = prev


def benchmark_cnot(
    num_qubits: int = 12,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    repeats: int = 5,
) -> dict[str, BenchmarkResult]:
    """Return dense vs. sparse timings for building and applying a single CNOT."""

    results: dict[str, BenchmarkResult] = {}
    control, target = 0, max(1, num_qubits - 1)
    dim = 2 ** num_qubits
    base_state = torch.randn(dim, dtype=dtype, device=device)
    base_state = base_state / torch.linalg.norm(base_state)

    for dense in (True, False):
        mode = "dense" if dense else "sparse"
        with _dense_flag(dense):
            build_time = _time_it(lambda: op.CNOT(control, target).build(num_qubits), repeats)
            gate = op.CNOT(control, target).build(num_qubits)

        def _apply_once():
            state = base_state.clone()
            with torch.no_grad():
                gate(state)

        apply_time = _time_it(_apply_once, repeats)
        results[mode] = BenchmarkResult(build_time=build_time, apply_time=apply_time)

    return results


def benchmark_ccnot(
    num_qubits: int = 12,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    repeats: int = 5,
) -> dict[str, BenchmarkResult]:
    """Return dense vs. sparse timings for building and applying a single CCNOT."""

    results: dict[str, BenchmarkResult] = {}
    controls = (0, 1)
    target = max(2, num_qubits - 1)
    dim = 2 ** num_qubits
    base_state = torch.randn(dim, dtype=dtype, device=device)
    base_state = base_state / torch.linalg.norm(base_state)

    for dense in (True, False):
        mode = "dense" if dense else "sparse"
        with _dense_flag(dense):
            build_time = _time_it(
                lambda: op.CCNOT(*controls, target=target).build(num_qubits), repeats
            )
            gate = op.CCNOT(*controls, target=target).build(num_qubits)

        def _apply_once():
            state = base_state.clone()
            with torch.no_grad():
                gate(state)

        apply_time = _time_it(_apply_once, repeats)
        results[mode] = BenchmarkResult(build_time=build_time, apply_time=apply_time)

    return results


__all__ = ["BenchmarkResult", "benchmark_cnot", "benchmark_ccnot"]
