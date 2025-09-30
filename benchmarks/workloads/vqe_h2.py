from __future__ import annotations

import math
import time
import torch

from benchmarks.accuracy.h2_reference import h2_ground_energy, h2_hamiltonian
from benchmarks.accuracy.metrics import energy_expectation
from benchmarks.ir.builders import hardware_efficient_ansatz
from benchmarks.memory_utils import get_peak_rss_mb
from benchmarks.workloads import WorkloadResult


def run_vqe_h2(backend, n_qubits: int, layers: int, seed: int) -> WorkloadResult:
    if n_qubits != 4:
        return WorkloadResult(
            execution_time_s=0.0,
            peak_memory_mb=get_peak_rss_mb(),
            accuracy_name="energy_abs_error",
            accuracy_value=math.nan,
            success=False,
            error="VQE only implemented at n=4",
        )

    torch.manual_seed(seed)
    gates, next_index = hardware_efficient_ansatz(n_qubits, layers)
    params = torch.randn(next_index, dtype=torch.float64)

    start = time.perf_counter()
    state = backend.simulate_state(n_qubits, gates, params, seed)
    execution_time = time.perf_counter() - start
    peak_mem = get_peak_rss_mb()

    hamiltonian = h2_hamiltonian()
    energy = energy_expectation(state, hamiltonian)
    target = h2_ground_energy(hamiltonian)
    accuracy = abs(float(energy.real.item()) - target)

    return WorkloadResult(
        execution_time_s=execution_time,
        peak_memory_mb=peak_mem,
        accuracy_name="energy_abs_error",
        accuracy_value=accuracy,
        success=True,
        error=None,
    )
