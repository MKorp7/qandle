from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import platform
from typing import Dict, Iterable, List

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

    vqe_circuit = utils.vqe_4q(args.vqe_depth, args.seed)
    qaoa_circuit = utils.qaoa_ring(args.qaoa_qubits, args.qaoa_layers, args.seed)
    circuits = [vqe_circuit, qaoa_circuit]

    run_id = 0
    try:
        for spec in iter_backends(device_keys):
            if not spec.is_available():
                print(f"Skipping {spec.key}: {spec.unavailable_reason or 'unknown'}")
                continue

            print(f"Running VQE/QAOA benchmarks for {spec.key}")

            for circuit in circuits:
                instance = spec.make(
                    n_qubits=circuit.n_qubits,
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
                        circuit,
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
                    "algorithm": circuit.algorithm,
                    "scenario": circuit.scenario,
                    "n_qubits": str(circuit.n_qubits),
                    "depth": str(circuit.depth),
                    "problem_id": circuit.problem_id,
                    "seed": str(args.seed),
                    "run_id": str(run_id),
                    "execution_time_s": f"{mean_time:.6f}",
                    "execution_time_std_s": f"{std_time:.6f}",
                    "reps": str(args.reps),
                    "peak_cpu_mb": f"{peak_cpu:.3f}",
                    "peak_gpu_mb": f"{peak_gpu:.3f}",
                    "success": str(success),
                    "error": error,
                    "tags": tags,
                }

                for metric_name, metric_value in metrics.items():
                    row = dict(base_row)
                    row["metric_name"] = metric_name
                    row["metric_value"] = str(metric_value)
                    writer.writerow(row)

                run_id += 1
    finally:
        handle.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VQE and QAOA benchmarks")
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
    parser.add_argument("--reps", type=int, default=5, help="Timing repetitions")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--out", type=str, default="vqe_qaoa.csv", help="Output CSV path")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline key")
    parser.add_argument("--tags", type=str, default="", help="Free-form tags")
    parser.add_argument("--vqe-depth", type=int, dest="vqe_depth", default=3, help="VQE ansatz depth")
    parser.add_argument("--qaoa-qubits", type=int, default=6, help="Number of qubits for QAOA")
    parser.add_argument("--qaoa-layers", type=int, default=3, help="Number of QAOA layers")

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.splitting = True if args.splitting == "on" else False
    run_benchmark(args)


if __name__ == "__main__":
    main()

