from __future__ import annotations

import argparse
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

from benchmarks import config as bench_config
from benchmarks.accuracy.metrics import state_fidelity
from benchmarks.framework_isolation import (
    import_pl_backend,
    import_qandle_new_backend,
    import_qandle_origin_backend,
)
from benchmarks.logging_utils import append_row
from benchmarks.workloads import WorkloadResult
from benchmarks.workloads.classifier import run_classifier
from benchmarks.workloads.qaoa_maxcut import run_qaoa_maxcut
from benchmarks.workloads.vqe_h2 import run_vqe_h2


ALGORITHMS = {"vqe_h2", "qaoa", "classifier"}
FRAMEWORKS = {"pennylane", "qadle_origin", "qandle_new"}


@dataclass
class WorkItem:
    framework: str
    algorithm: str
    n_qubits: int
    depth_or_p: int
    problem_id: str
    seed: int
    run_id: int


def parse_list(value: str, default: Iterable) -> List[str]:
    if not value or value.lower() == "all":
        return [str(item).lower() for item in default]
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_int_list(value: str, default: Iterable[int]) -> List[int]:
    if not value or value.lower() == "all":
        return list(default)
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def load_backend(framework: str, qandle_new_path: str):
    if framework == "pennylane":
        return import_pl_backend()
    if framework == "qadle_origin":
        return import_qandle_origin_backend()
    if framework == "qandle_new":
        return import_qandle_new_backend(qandle_new_path)
    raise ValueError(f"Unknown framework {framework}")


def run_algorithm(backend, item: WorkItem) -> WorkloadResult:
    if item.algorithm == "vqe_h2":
        return run_vqe_h2(backend, item.n_qubits, item.depth_or_p, item.seed)
    if item.algorithm == "qaoa":
        return run_qaoa_maxcut(backend, item.n_qubits, item.depth_or_p, item.seed)
    if item.algorithm == "classifier":
        return run_classifier(backend, item.n_qubits, item.depth_or_p, item.seed)
    raise ValueError(f"Unknown algorithm {item.algorithm}")


def _worker_entry(item: WorkItem, qandle_new_path: str) -> Dict[str, object]:
    try:
        random.seed(item.seed)
        np.random.seed(item.seed)
        torch.manual_seed(item.seed)

        backend = load_backend(item.framework, qandle_new_path)
        result = run_algorithm(backend, item)
        row: Dict[str, object] = {
            "framework": item.framework,
            "algorithm": item.algorithm,
            "n_qubits": item.n_qubits,
            "depth_or_p": item.depth_or_p,
            "problem_id": item.problem_id,
            "seed": item.seed,
            "run_id": item.run_id,
            "execution_time_s": float(result.execution_time_s),
            "peak_memory_mb": float(result.peak_memory_mb),
            "accuracy_name": result.accuracy_name,
            "accuracy_value": float(result.accuracy_value)
            if isinstance(result.accuracy_value, (int, float))
            else result.accuracy_value,
            "success": result.success,
            "error": result.error or "",
        }
    except Exception as exc:  # pragma: no cover - defensive
        row = {
            "framework": item.framework,
            "algorithm": item.algorithm,
            "n_qubits": item.n_qubits,
            "depth_or_p": item.depth_or_p,
            "problem_id": item.problem_id,
            "seed": item.seed,
            "run_id": item.run_id,
            "execution_time_s": 0.0,
            "peak_memory_mb": math.nan,
            "accuracy_name": "",
            "accuracy_value": math.nan,
            "success": False,
            "error": str(exc)[:200],
        }
    return row


def _worker_smoke_state(framework: str, qandle_new_path: str):
    item = WorkItem(
        framework=framework,
        algorithm="qaoa",
        n_qubits=4,
        depth_or_p=1,
        problem_id="ring_4",
        seed=bench_config.DEFAULT_CONFIG.seed,
        run_id=-1,
    )
    random.seed(item.seed)
    np.random.seed(item.seed)
    torch.manual_seed(item.seed)
    backend = load_backend(framework, qandle_new_path)
    from benchmarks.workloads.qaoa_maxcut import build_ring_edges
    from benchmarks.ir.builders import qaoa_layers_ir

    edges = build_ring_edges(item.n_qubits)
    gates, next_index = qaoa_layers_ir(edges, item.depth_or_p, item.n_qubits)
    params = torch.zeros(next_index, dtype=torch.float64)
    params[0] = 2.0 * 0.7
    params[1] = 2.0 * 0.5
    state = backend.simulate_state(item.n_qubits, gates, params, item.seed)
    return framework, state


def build_work_items(cfg: bench_config.BenchmarkConfig) -> List[WorkItem]:
    items: List[WorkItem] = []
    run_id = 0
    for algorithm in cfg.algorithms:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unsupported algorithm {algorithm}")
        for framework in cfg.frameworks:
            if framework not in FRAMEWORKS:
                raise ValueError(f"Unsupported framework {framework}")
            for n in cfg.n_list:
                for depth in cfg.depth_list:
                    problem_id = "h2" if algorithm == "vqe_h2" else (
                        f"ring_{n}" if algorithm == "qaoa" else "moons"
                    )
                    items.append(
                        WorkItem(
                            framework=framework,
                            algorithm=algorithm,
                            n_qubits=n,
                            depth_or_p=depth,
                            problem_id=problem_id,
                            seed=cfg.seed,
                            run_id=run_id,
                        )
                    )
                    run_id += 1
    return items


def run_controller(cfg: bench_config.BenchmarkConfig) -> None:
    qandle_new_path = os.environ.get("QANDLE_NEW_PATH")
    if not qandle_new_path:
        repo_root = Path(__file__).resolve().parent.parent
        qandle_new_path = str(repo_root)
        os.environ["QANDLE_NEW_PATH"] = qandle_new_path

    work_items = build_work_items(cfg)

    # Handle VQE skips for n != 4 upfront
    filtered_items: List[WorkItem] = []
    for item in work_items:
        if item.algorithm == "vqe_h2" and item.n_qubits != 4:
            append_row(
                cfg.output_csv,
                {
                    "framework": item.framework,
                    "algorithm": item.algorithm,
                    "n_qubits": item.n_qubits,
                    "depth_or_p": item.depth_or_p,
                    "problem_id": item.problem_id,
                    "seed": item.seed,
                    "run_id": item.run_id,
                    "execution_time_s": 0.0,
                    "peak_memory_mb": math.nan,
                    "accuracy_name": "energy_abs_error",
                    "accuracy_value": math.nan,
                    "success": False,
                    "error": "VQE only implemented at n=4",
                },
            )
        else:
            filtered_items.append(item)

    if not filtered_items:
        return

    max_workers = cfg.max_workers or (min(4, os.cpu_count() or 1) if cfg.parallel else 1)
    if not cfg.parallel:
        max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_entry, item, qandle_new_path): item for item in filtered_items
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            item = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover
                row = {
                    "framework": item.framework,
                    "algorithm": item.algorithm,
                    "n_qubits": item.n_qubits,
                    "depth_or_p": item.depth_or_p,
                    "problem_id": item.problem_id,
                    "seed": item.seed,
                    "run_id": item.run_id,
                    "execution_time_s": 0.0,
                    "peak_memory_mb": math.nan,
                    "accuracy_name": "",
                    "accuracy_value": math.nan,
                    "success": False,
                    "error": str(exc)[:200],
                }
            append_row(cfg.output_csv, row)

    if cfg.quick_smoke:
        with ProcessPoolExecutor(max_workers=len(cfg.frameworks)) as executor:
            futures = [executor.submit(_worker_smoke_state, fw, qandle_new_path) for fw in cfg.frameworks]
            states = dict(f.result() for f in futures)
        frameworks = list(states.keys())
        for i in range(len(frameworks)):
            for j in range(i + 1, len(frameworks)):
                fi, fj = frameworks[i], frameworks[j]
                fidelity = state_fidelity(states[fi], states[fj])
                if fidelity < 1 - 1e-6:
                    print(
                        f"[quick-smoke] Warning: fidelity between {fi} and {fj} is {fidelity:.6f}",
                        flush=True,
                    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum framework benchmarking suite")
    parser.add_argument("--algorithms", default="all", help="Algorithms to run (comma separated or 'all')")
    parser.add_argument("--frameworks", default="all", help="Frameworks to benchmark (comma separated or 'all')")
    parser.add_argument("--n-list", default="all", help="Comma separated qubit counts")
    parser.add_argument("--depth-list", default="all", help="Comma separated depths or QAOA p values")
    parser.add_argument("--seed", type=int, default=bench_config.DEFAULT_CONFIG.seed, help="Global random seed")
    parser.add_argument("--output", default=bench_config.DEFAULT_CONFIG.output_csv, help="CSV output path")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum worker processes")
    parser.add_argument("--quick-smoke", action="store_true", help="Run a reduced smoke test set")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = bench_config.BenchmarkConfig()
    cfg.output_csv = args.output
    cfg.parallel = args.parallel
    cfg.max_workers = args.max_workers
    cfg.quick_smoke = args.quick_smoke
    cfg.seed = args.seed

    if args.quick_smoke:
        cfg.algorithms = sorted(ALGORITHMS)
        cfg.frameworks = sorted(FRAMEWORKS)
        cfg.n_list = [4]
        cfg.depth_list = [1]
    else:
        cfg.algorithms = parse_list(args.algorithms, bench_config.DEFAULT_CONFIG.algorithms)
        cfg.frameworks = parse_list(args.frameworks, bench_config.DEFAULT_CONFIG.frameworks)
        cfg.n_list = parse_int_list(args.n_list, bench_config.DEFAULT_CONFIG.n_list)
        cfg.depth_list = parse_int_list(args.depth_list, bench_config.DEFAULT_CONFIG.depth_list)

    run_controller(cfg)


if __name__ == "__main__":
    main()
