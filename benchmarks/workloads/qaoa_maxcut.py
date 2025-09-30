from __future__ import annotations

import time
from typing import List, Tuple

import networkx as nx
import torch

from benchmarks.accuracy.graph_optimum import optimal_maxcut
from benchmarks.accuracy.metrics import expected_maxcut_from_state
from benchmarks.ir.builders import qaoa_layers_ir
from benchmarks.memory_utils import get_peak_rss_mb
from benchmarks.workloads import WorkloadResult


def build_ring_edges(n_qubits: int) -> List[Tuple[int, int]]:
    graph = nx.cycle_graph(n_qubits)
    edges = sorted((int(a), int(b)) for a, b in graph.edges())
    return edges


def run_qaoa_maxcut(backend, n_qubits: int, p: int, seed: int) -> WorkloadResult:
    edges = build_ring_edges(n_qubits)
    gates, next_index = qaoa_layers_ir(edges, p, n_qubits)

    params = torch.zeros(next_index, dtype=torch.float64)
    for layer in range(p):
        gamma_angle = 2.0 * 0.7
        beta_angle = 2.0 * 0.5
        params[2 * layer] = gamma_angle
        params[2 * layer + 1] = beta_angle

    start = time.perf_counter()
    state = backend.simulate_state(n_qubits, gates, params, seed)
    execution_time = time.perf_counter() - start
    peak_mem = get_peak_rss_mb()

    expected = expected_maxcut_from_state(state, edges, n_qubits)
    optimum = optimal_maxcut(n_qubits, edges)
    accuracy_value = float(optimum - expected.real.item())

    return WorkloadResult(
        execution_time_s=execution_time,
        peak_memory_mb=peak_mem,
        accuracy_name="gap_to_optimum",
        accuracy_value=accuracy_value,
        success=True,
        error=None,
    )
