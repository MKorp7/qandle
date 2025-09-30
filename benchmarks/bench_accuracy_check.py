from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import platform
from typing import Dict, Iterable, List, Tuple

from benchmarks.backends import available_backends, iter_backends
from benchmarks import utils


FIELDNAMES = [
    "timestamp_iso",
    "host",
    "framework",
    "family",
    "device_key",
    "label",
    "sim_type",
    "hardware",
    "dtype",
    "threads",
    "bond_dim",
    "splitting",
    "shots",
    "algorithm",
    "scenario",
    "n_qubits",
    "depth",
    "problem_id",
    "seed",
    "run_id",
    "execution_time_s",
    "execution_time_std_s",
    "reps",
    "peak_cpu_mb",
    "peak_gpu_mb",
    "metric_name",
    "metric_value",
    "success",
    "error",
    "tags",
]


THRESHOLDS: Dict[str, Tuple[str, float, str]] = {
    "ghz": ("state_fidelity", 0.99, "ge"),
    "single_qubit_ry": ("expval_abs_error", 1e-5, "le"),
    "probabilities": ("probs_tv_distance", 1e-5, "le"),
}


def parse_devices(value: str | None) -> List[str]:
    if not value:
        return available_backends()
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_writer(path: str) -> tuple[csv.DictWriter, any]:
    exists = os.path.exists(path)
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if not exists:
        writer.writeheader()
    return writer, handle


def run_benchmark(args: argparse.Namespace) -> None:
    device_keys = parse_devices(args.devices)
    if not device_keys:
        raise SystemExit("No devices selected (or available) for benchmarking.")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception:
        print("CUDA available: unknown (torch not installed)")

    utils.stable_seed(args.seed)
    utils.configure_threads(args.threads)

    writer, handle = ensure_writer(args.out)
    host = platform.node()
    tags = args.tags or ""

    tests = [
        utils.ghz_spec(args.seed),
        *utils.ry_expectation_specs([0.0, 0.2, 0.4, 0.6]),
        utils.probability_tv_spec(args.seed),
    ]

    run_id = 0
    try:
        for spec in iter_backends(device_keys):
            if not spec.is_available():
                print(f"Skipping {spec.key}: {spec.unavailable_reason or 'unknown'}")
                continue

            print(f"Running accuracy checks for {spec.key}")

            for test in tests:
                instance = spec.make(
                    n_qubits=test.n_qubits,
                    shots=utils.shots_value(args.shots),
                    dtype=args.dtype or spec.defaults.get("dtype"),
                    threads=args.threads,
                    bond_dim=args.bond_dim,
                    splitting=args.splitting,
                )

                dtype = utils.numpy_dtype(instance.dtype, spec.defaults.get("dtype", "complex128"))
                shots = utils.shots_value(args.shots)

                try:
                    metrics, mean_time, std_time, peak_cpu, peak_gpu = utils.execute_circuit(
                        test,
                        dtype,
                        shots,
                        spec.hardware,
                        args.reps,
                        args.seed + run_id,
                    )
                    success = True
                    error = ""
                except Exception as exc:  # pragma: no cover
                    metrics = {"metric": float("nan")}
                    mean_time = std_time = peak_cpu = peak_gpu = 0.0
                    success = False
                    error = str(exc)

                if not metrics:
                    metrics = {"metric": float("nan")}

                metric_name, threshold, relation = THRESHOLDS.get(
                    test.scenario, (next(iter(metrics.keys())), float("nan"), "ge")
                )

                value = metrics.get(metric_name, float("nan"))
                comparison_success = success
                comparison_error = error
                if success:
                    if relation == "ge":
                        comparison_success = bool(value >= threshold)
                    else:
                        comparison_success = bool(value <= threshold)
                    if not comparison_success:
                        comparison_error = f"{metric_name} threshold {threshold} not met"

                base_row: Dict[str, str] = {
                    "timestamp_iso": _dt.datetime.utcnow().isoformat(),
                    "host": host,
                    "framework": spec.family,
                    "family": spec.family,
                    "device_key": spec.key,
                    "label": spec.label,
                    "sim_type": spec.sim_type,
                    "hardware": spec.hardware,
                    "dtype": str(instance.dtype or ""),
                    "threads": str(instance.threads or ""),
                    "bond_dim": str(instance.bond_dim or ""),
                    "splitting": utils.format_bool_flag(instance.splitting),
                    "shots": str(shots),
                    "algorithm": test.algorithm,
                    "scenario": test.scenario,
                    "n_qubits": str(test.n_qubits),
                    "depth": str(test.depth),
                    "problem_id": test.problem_id,
                    "seed": str(args.seed),
                    "run_id": str(run_id),
                    "execution_time_s": f"{mean_time:.6f}",
                    "execution_time_std_s": f"{std_time:.6f}",
                    "reps": str(args.reps),
                    "peak_cpu_mb": f"{peak_cpu:.3f}",
                    "peak_gpu_mb": f"{peak_gpu:.3f}",
                    "metric_name": metric_name,
                    "metric_value": str(value),
                    "success": str(comparison_success),
                    "error": comparison_error,
                    "tags": tags,
                }
                writer.writerow(base_row)
                run_id += 1
    finally:
        handle.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Accuracy benchmark")
    parser.add_argument("--devices", type=str, default=None, help="Comma separated backend keys")
    parser.add_argument("--dtype", type=str, default=None, help="Preferred dtype")
    parser.add_argument("--threads", type=int, default=None, help="Override CPU threads")
    parser.add_argument("--bond-dim", type=int, dest="bond_dim", default=None, help="MPS bond dimension")
    parser.add_argument(
        "--splitting",
        choices=["on", "off"],
        default="off",
        help="Enable memory splitting where supported",
    )
    parser.add_argument("--shots", type=int, default=0, help="Shot count (0 for analytic)")
    parser.add_argument("--reps", type=int, default=3, help="Timing repetitions")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--out", type=str, default="accuracy.csv", help="Output CSV path")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline key")
    parser.add_argument("--tags", type=str, default="", help="Free-form tags")

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.splitting = True if args.splitting == "on" else False
    run_benchmark(args)


if __name__ == "__main__":
    main()

