"""Utility helpers shared across benchmark scripts.

The functions in this module are intentionally self-contained to minimise
third-party dependencies.  The benchmark suite relies on them to
construct deterministic circuit specifications, execute lightweight
simulations for sanity/accuracy checks, and implement repeatable timing
loops with memory instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from statistics import mean, pstdev
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency used for CUDA synchronisation / memory checks
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


Operation = Tuple[str, Tuple[int, ...], Tuple[float, ...]]


@dataclass
class CircuitSpec:
    algorithm: str
    scenario: str
    n_qubits: int
    depth: int
    problem_id: str
    operations: List[Operation]
    measure: Callable[[np.ndarray, int, np.random.Generator], Dict[str, float]]


@dataclass
class ParametricCircuitSpec:
    algorithm: str
    scenario: str
    n_qubits: int
    depth: int
    problem_id: str
    param_shape: Tuple[int, ...]
    build_operations: Callable[[np.ndarray], List[Operation]]
    observable: Callable[[np.ndarray], float]
    init_params: Callable[[np.random.Generator], np.ndarray]


DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def stable_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:  # pragma: no cover - safeguard
        pass
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def numpy_dtype(dtype: Optional[str], fallback: str) -> np.dtype:
    key = dtype or fallback
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{key}'. Supported: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[key]


def configure_threads(threads: Optional[int]) -> None:
    if threads and _TORCH_AVAILABLE:
        torch.set_num_threads(int(threads))


def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    if axis == "x":
        return np.array(
            [
                [math.cos(theta / 2), -1j * math.sin(theta / 2)],
                [-1j * math.sin(theta / 2), math.cos(theta / 2)],
            ],
            dtype=np.complex128,
        )
    if axis == "y":
        return np.array(
            [
                [math.cos(theta / 2), -math.sin(theta / 2)],
                [math.sin(theta / 2), math.cos(theta / 2)],
            ],
            dtype=np.complex128,
        )
    if axis == "z":
        return np.array(
            [
                [math.cos(theta / 2) - 1j * math.sin(theta / 2), 0],
                [0, math.cos(theta / 2) + 1j * math.sin(theta / 2)],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported rotation axis '{axis}'")


def hadamard() -> np.ndarray:
    return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, target, -1)
    state = np.tensordot(state, gate.T, axes=([-1], [0]))
    state = np.moveaxis(state, -1, target)
    return state.reshape(-1)


def apply_operation(state: np.ndarray, op: Operation, n_qubits: int) -> np.ndarray:
    name, targets, params = op
    if name == "H":
        return apply_single_qubit_gate(state, hadamard(), targets[0], n_qubits)
    if name == "RX":
        return apply_single_qubit_gate(state, rotation_matrix("x", params[0]), targets[0], n_qubits)
    if name == "RY":
        return apply_single_qubit_gate(state, rotation_matrix("y", params[0]), targets[0], n_qubits)
    if name == "RZ":
        return apply_single_qubit_gate(state, rotation_matrix("z", params[0]), targets[0], n_qubits)
    if name == "CNOT":
        control, target = targets
        if control == target:
            raise ValueError("control and target must differ for CNOT")
        mask_control = 1 << control
        mask_target = 1 << target
        new_state = state.copy()
        for basis_index in range(state.size):
            if basis_index & mask_control:
                flipped = basis_index ^ mask_target
                if flipped > basis_index:
                    new_state[basis_index], new_state[flipped] = state[flipped], state[basis_index]
        return new_state
    if name == "CZ":
        control, target = targets
        mask = (1 << control) | (1 << target)
        new_state = state.copy()
        for basis_index in range(state.size):
            if (basis_index & mask) == mask:
                new_state[basis_index] = -state[basis_index]
        return new_state
    raise ValueError(f"Unsupported operation {name}")


def simulate_statevector(n_qubits: int, operations: Sequence[Operation], dtype: np.dtype) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=dtype)
    state[0] = 1.0
    for op in operations:
        state = apply_operation(state, op, n_qubits)
    return state


def expectation_z(state: np.ndarray, n_qubits: int, wire: int) -> float:
    prob0 = probability_of_bit(state, n_qubits, wire, value=0)
    prob1 = probability_of_bit(state, n_qubits, wire, value=1)
    return float(prob0 - prob1)


def probability_of_bit(state: np.ndarray, n_qubits: int, wire: int, value: int) -> float:
    probs = np.abs(state) ** 2
    total = 0.0
    for index, amplitude in enumerate(probs):
        if ((index >> wire) & 1) == value:
            total += float(amplitude)
    return total


def probabilities(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def fidelity(state: np.ndarray, target: np.ndarray) -> float:
    inner = np.vdot(target, state)
    return float(np.abs(inner) ** 2)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def run_timed(fn: Callable[[], Dict[str, float]], hardware: str, reps: int) -> Tuple[Dict[str, float], float, float, float, float]:
    """Execute ``fn`` with warm-up and gather timing/memory statistics."""

    results: Dict[str, float] = {}
    timings: List[Tuple[float, float, float]] = []
    for rep in range(reps + 1):
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        import tracemalloc

        tracemalloc.start()
        start = time.perf_counter()
        current_result = fn()
        if hardware.upper() == "GPU" and _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            gpu_peak = 0.0
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_peak = peak / (1024**2)
        if rep > 0:
            timings.append((end - start, cpu_peak, gpu_peak))
            results = current_result

    if not timings:
        return results, 0.0, 0.0, 0.0, 0.0

    time_values = [t[0] for t in timings]
    cpu_peaks = [t[1] for t in timings]
    gpu_peaks = [t[2] for t in timings]
    mean_time = mean(time_values)
    std_time = pstdev(time_values) if len(time_values) > 1 else 0.0
    return results, mean_time, std_time, max(cpu_peaks), max(gpu_peaks)


def scaling_basic(n_qubits: int, depth: int, seed: int) -> CircuitSpec:
    ops: List[Operation] = []
    rng = np.random.default_rng(seed)
    for _ in range(depth):
        for qubit in range(n_qubits):
            angle = rng.uniform(0, 2 * math.pi)
            ops.append(("RY", (qubit,), (angle,)))
        for control, target in zip(range(n_qubits - 1), range(1, n_qubits)):
            ops.append(("CNOT", (control, target), ()))
    return CircuitSpec(
        algorithm="qubit_scaling",
        scenario="runtime",
        n_qubits=n_qubits,
        depth=depth,
        problem_id=f"scaling_{n_qubits}_{depth}",
        operations=ops,
        measure=lambda state, shots, rng: {"state_norm": float(np.linalg.norm(state))},
    )


def depth_basic(n_qubits: int, depth: int, seed: int) -> CircuitSpec:
    ops: List[Operation] = []
    rng = np.random.default_rng(seed)
    for layer in range(depth):
        for qubit in range(n_qubits):
            axis = "x" if layer % 2 == 0 else "y"
            angle = rng.uniform(0, 2 * math.pi)
            ops.append((f"R{axis.upper()}", (qubit,), (angle,)))
        for control, target in zip(range(n_qubits - 1), range(1, n_qubits)):
            ops.append(("CNOT", (control, target), ()))
    return CircuitSpec(
        algorithm="depth_scaling",
        scenario="runtime",
        n_qubits=n_qubits,
        depth=depth,
        problem_id=f"depth_{n_qubits}_{depth}",
        operations=ops,
        measure=lambda state, shots, rng: {"state_norm": float(np.linalg.norm(state))},
    )


def grad_basic(n_qubits: int, depth: int, seed: int) -> ParametricCircuitSpec:
    rng = np.random.default_rng(seed)
    param_count = n_qubits * depth

    def _build(params: np.ndarray) -> List[Operation]:
        ops: List[Operation] = []
        param_iter = iter(params.flatten())
        for _ in range(depth):
            for qubit in range(n_qubits):
                angle = next(param_iter)
                ops.append(("RY", (qubit,), (angle,)))
            for control, target in zip(range(n_qubits - 1), range(1, n_qubits)):
                ops.append(("CNOT", (control, target), ()))
        return ops

    def _init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0, 2 * math.pi, size=param_count).reshape((depth, n_qubits))

    def _observable(state: np.ndarray) -> float:
        return expectation_z(state, n_qubits, n_qubits - 1)

    return ParametricCircuitSpec(
        algorithm="gradient_perf",
        scenario="gradient",
        n_qubits=n_qubits,
        depth=depth,
        problem_id=f"grad_{n_qubits}_{depth}",
        param_shape=(depth, n_qubits),
        build_operations=_build,
        observable=_observable,
        init_params=_init,
    )


def vqe_4q(depth: int, seed: int) -> CircuitSpec:
    n_qubits = 4
    rng = np.random.default_rng(seed)
    ops: List[Operation] = []
    for _ in range(depth):
        for q in range(n_qubits):
            ops.append(("RY", (q,), (rng.uniform(0, 2 * math.pi),)))
        for control, target in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            ops.append(("CNOT", (control, target), ()))

    def measure(state: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, float]:
        total = 0.0
        for q in range(n_qubits):
            total += expectation_z(state, n_qubits, q)
        return {"expval_z_sum": total}

    return CircuitSpec(
        algorithm="vqe",
        scenario="runtime",
        n_qubits=n_qubits,
        depth=depth,
        problem_id=f"vqe4q_{depth}",
        operations=ops,
        measure=measure,
    )


def qaoa_ring(n_qubits: int, p: int, seed: int) -> CircuitSpec:
    rng = np.random.default_rng(seed)
    ops: List[Operation] = []
    for layer in range(p):
        gamma = rng.uniform(0, math.pi)
        beta = rng.uniform(0, math.pi)
        for i in range(n_qubits):
            ops.append(("H", (i,), ()))
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            ops.append(("CNOT", (i, j), ()))
            ops.append(("RZ", (j,), (gamma,)))
            ops.append(("CNOT", (i, j), ()))
        for i in range(n_qubits):
            ops.append(("RX", (i,), (beta,)))

    def measure(state: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, float]:
        total = 0.0
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            total += expectation_z_pair(state, n_qubits, i, j)
        return {"zz_ring": total}

    return CircuitSpec(
        algorithm="qaoa",
        scenario="runtime",
        n_qubits=n_qubits,
        depth=p,
        problem_id=f"qaoa_ring_{n_qubits}_{p}",
        operations=ops,
        measure=measure,
    )


def expectation_z_pair(state: np.ndarray, n_qubits: int, wire_a: int, wire_b: int) -> float:
    probs = probabilities(state)
    total = 0.0
    for index, prob in enumerate(probs):
        bit_a = (index >> wire_a) & 1
        bit_b = (index >> wire_b) & 1
        value = (1 if bit_a == 0 else -1) * (1 if bit_b == 0 else -1)
        total += value * prob
    return float(total)


def ghz_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1 / math.sqrt(2)
    state[-1] = 1 / math.sqrt(2)
    return state


def ghz_spec(seed: int = 0) -> CircuitSpec:
    n_qubits = 3
    ops: List[Operation] = [("H", (0,), ())]
    for target in range(1, n_qubits):
        ops.append(("CNOT", (0, target), ()))

    target_state = ghz_state(n_qubits)

    def measure(state: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, float]:
        return {"state_fidelity": fidelity(state, target_state)}

    return CircuitSpec(
        algorithm="accuracy",
        scenario="ghz",
        n_qubits=n_qubits,
        depth=2,
        problem_id="ghz3",
        operations=ops,
        measure=measure,
    )


def ry_expectation_specs(thetas: Sequence[float]) -> List[CircuitSpec]:
    specs: List[CircuitSpec] = []
    for idx, theta in enumerate(thetas):
        ops: List[Operation] = [("RY", (0,), (theta,))]

        def measure(theta_value: float) -> Callable[[np.ndarray, int, np.random.Generator], Dict[str, float]]:
            def _inner(state: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, float]:
                expectation = expectation_z(state, 1, 0)
                expected = math.cos(theta_value)
                return {"expval_abs_error": abs(expectation - expected)}

            return _inner

        specs.append(
            CircuitSpec(
                algorithm="accuracy",
                scenario="single_qubit_ry",
                n_qubits=1,
                depth=1,
                problem_id=f"ry_{idx}",
                operations=list(ops),
                measure=measure(theta),
            )
        )
    return specs


def probability_tv_spec(seed: int = 0) -> CircuitSpec:
    n_qubits = 2
    ops: List[Operation] = [("H", (0,), ()), ("CNOT", (0, 1), ())]
    target = probabilities(simulate_statevector(n_qubits, ops, np.complex128))

    def measure(state: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, float]:
        probs = probabilities(state)
        return {"probs_tv_distance": total_variation_distance(probs, target)}

    return CircuitSpec(
        algorithm="accuracy",
        scenario="probabilities",
        n_qubits=n_qubits,
        depth=1,
        problem_id="prob_tv",
        operations=ops,
        measure=measure,
    )


def param_shift_gradient(
    spec: ParametricCircuitSpec,
    params: np.ndarray,
    dtype: np.dtype,
) -> Tuple[np.ndarray, float]:
    flat_params = params.flatten()
    gradient = np.zeros_like(flat_params)

    def evaluate(current: np.ndarray) -> float:
        ops = spec.build_operations(current)
        state = simulate_statevector(spec.n_qubits, ops, dtype)
        return spec.observable(state)

    for idx in range(flat_params.size):
        shift = np.zeros_like(flat_params)
        shift[idx] = math.pi / 2
        plus = flat_params + shift
        minus = flat_params - shift
        gradient[idx] = (evaluate(plus) - evaluate(minus)) / 2.0

    gradient = gradient.reshape(spec.param_shape)
    grad_norm = float(np.linalg.norm(gradient))
    return gradient, grad_norm


def execute_circuit(
    circuit: CircuitSpec,
    dtype: np.dtype,
    shots: int,
    hardware: str,
    reps: int,
    seed: int,
) -> Tuple[Dict[str, float], float, float, float, float]:
    rng = np.random.default_rng(seed)

    def _runner() -> Dict[str, float]:
        state = simulate_statevector(circuit.n_qubits, circuit.operations, dtype)
        return circuit.measure(state, shots, rng)

    return run_timed(_runner, hardware, reps)


def execute_parametric_circuit(
    circuit: ParametricCircuitSpec,
    dtype: np.dtype,
    hardware: str,
    reps: int,
    seed: int,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    params = circuit.init_params(rng)

    def _runner() -> Dict[str, float]:
        gradient, grad_norm = param_shift_gradient(circuit, params, dtype)
        return {"grad_norm": grad_norm}

    metrics, mean_time, std_time, cpu_peak, gpu_peak = run_timed(_runner, hardware, reps)
    gradient, grad_norm = param_shift_gradient(circuit, params, dtype)
    metrics["grad_norm"] = grad_norm
    return gradient, grad_norm, mean_time, std_time, cpu_peak, gpu_peak


def format_bool_flag(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "on" if value else "off"


def shots_value(shots: int) -> int:
    return max(int(shots), 0)


__all__ = [
    "CircuitSpec",
    "ParametricCircuitSpec",
    "configure_threads",
    "execute_circuit",
    "execute_parametric_circuit",
    "format_bool_flag",
    "grad_basic",
    "numpy_dtype",
    "param_shift_gradient",
    "probability_tv_spec",
    "qaoa_ring",
    "run_timed",
    "scaling_basic",
    "shots_value",
    "stable_seed",
    "vqe_4q",
    "ghz_spec",
    "ry_expectation_specs",
]

