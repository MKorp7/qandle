"""Command line entry point for the benchmarking suite."""

from __future__ import annotations

import argparse
import csv
import json
import math

import multiprocessing
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks import frameworks, plots, specs

__all__ = ["main"]


try:  # pragma: no cover - platform specific import
    import resource  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None  # type: ignore[assignment]

_RESOURCE_SCALE = 1024.0 if sys.platform == "darwin" else 1.0


class _MemoryTracker:
    """Cross-platform helper to estimate peak memory usage for a worker."""

    def __init__(self) -> None:
        self._baseline = math.nan
        self._psutil_process = None
        self._psutil_baseline_peak = math.nan
        self._psutil_baseline_rss = math.nan
        self._using_resource = resource is not None

    def start(self) -> None:
        if self._using_resource:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[arg-type]
            self._baseline = float(usage) / _RESOURCE_SCALE
            return
        try:  # pragma: no cover - optional dependency path
            import psutil  # type: ignore[import]
        except ImportError:
            self._using_resource = False
            return
        self._psutil_process = psutil.Process()
        info = self._psutil_process.memory_info()
        self._psutil_baseline_peak = float(_extract_peak_memory(info))
        self._psutil_baseline_rss = float(getattr(info, "rss", 0.0))

    def stop(self) -> float:
        if self._using_resource:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[arg-type]
            current = float(usage) / _RESOURCE_SCALE
            return max(current - self._baseline, 0.0)
        if self._psutil_process is None:
            return float("nan")
        info = self._psutil_process.memory_info()
        peak_bytes = float(_extract_peak_memory(info))
        baseline = self._psutil_baseline_peak
        if math.isnan(peak_bytes) or math.isnan(baseline):
            peak_bytes = float(getattr(info, "rss", 0.0))
            baseline = self._psutil_baseline_rss
        return max((peak_bytes - baseline) / 1024.0, 0.0)


def _extract_peak_memory(info: object) -> float:
    for attr in ("peak_wset", "peak_rss", "rss"):
        value = getattr(info, attr, None)
        if value is not None:
            return float(value)
    return float("nan")



def _worker(entry):
    spec_dict, framework_name, options_dict = entry
    from scripts.benchmarks.specs import CircuitSpec
    from scripts.benchmarks.frameworks import run_framework_once

    spec = CircuitSpec(**spec_dict)
    tracker = _MemoryTracker()
    tracker.start()

    start_time = time.perf_counter()
    try:
        details = run_framework_once(framework_name, spec, options_dict)
        status = "ok"
    except Exception as exc:  # pragma: no cover - executed in subprocess
        status = "error"
        details = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        runtime = time.perf_counter() - start_time
        peak_mem = tracker.stop()

    return {
        "status": status,
        "details": details,
        "runtime_s": runtime,
        "peak_memory_kb": peak_mem,
    }


def _process_target(queue: multiprocessing.Queue, spec_dict, framework_name: str, options_dict: Dict[str, object]):
    payload = _worker((spec_dict, framework_name, options_dict))
    queue.put(payload)


def _execute_benchmark(spec: specs.CircuitSpec, framework_name: str, options: Dict[str, object], run_id: int):
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[assignment]
    process = ctx.Process(
        target=_process_target,
        args=(queue, spec.to_dict(), framework_name, options),
    )
    process.start()
    process.join()
    payload = queue.get() if not queue.empty() else {
        "status": "error",
        "details": {"error": "No result returned"},
        "runtime_s": float("nan"),
        "peak_memory_kb": 0.0,
    }
    queue.close()
    queue.join_thread()
    if process.exitcode not in (0, None) and payload.get("status") == "ok":
        payload = {
            "status": "error",
            "details": {
                "error": f"Process exited with status {process.exitcode}",
            },
            "runtime_s": payload.get("runtime_s", float("nan")),
            "peak_memory_kb": payload.get("peak_memory_kb", float("nan")),
        }
    payload.update(
        {
            "framework": framework_name,
            "circuit": spec.name,
            "category": spec.category,
            "num_qubits": spec.num_qubits,
            "depth": spec.depth,
            "run": run_id,
        }
    )
    return payload


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})


def _summarise(results: List[Dict[str, object]]):
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in results:
        key = (row["framework"], row["circuit"])
        grouped[key].append(row)
    summary = []
    for (framework_name, circuit), rows in grouped.items():
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        if not ok_rows:
            continue
        runtime = sum(float(r["runtime_s"]) for r in ok_rows) / len(ok_rows)
        memory = sum(float(r["peak_memory_kb"]) for r in ok_rows) / len(ok_rows)
        summary.append(
            {
                "framework": framework_name,
                "circuit": circuit,
                "category": ok_rows[0]["category"],
                "num_qubits": ok_rows[0]["num_qubits"],
                "depth": ok_rows[0]["depth"],
                "avg_runtime_s": runtime,
                "avg_peak_memory_kb": memory,
                "success_runs": len(ok_rows),
                "total_runs": len(rows),
            }
        )
    return summary


def _parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="Run Qandle benchmark suite")
    parser.add_argument("--preset", choices=sorted(specs.PRESETS), help="Use a predefined benchmark preset")
    parser.add_argument("--circuits", nargs="*", help="Subset of circuits to execute")
    parser.add_argument("--frameworks", nargs="*", help="Subset of frameworks to execute")
    parser.add_argument("--repetitions", type=int, help="Number of repetitions per configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/benchmarks/results"))
    parser.add_argument("--no-plots", action="store_true", help="Do not generate plot outputs")
    parser.add_argument("--legacy-site-packages", type=str, help="Path to site-packages containing legacy Qandle")
    parser.add_argument("--legacy-auto-install", action="store_true", help="Automatically download the legacy release if needed")
    parser.add_argument("--legacy-version", type=str, default="0.1.12", help="Legacy Qandle version to install when auto-install is enabled")
    parser.add_argument("--legacy-cache-dir", type=str, help="Directory used to cache auto-installed legacy packages")
    parser.add_argument("--dtype", type=str, default="float32", help="Torch dtype for random parameters")
    parser.add_argument("--mps-max-bond", type=int, default=128, help="Bond dimension for MPS backend")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    preset = specs.resolve_preset(getattr(args, "preset", None))
    circuits = args.circuits or preset.get("circuits") or tuple(specs.CIRCUIT_LIBRARY)
    frameworks_to_run = args.frameworks or preset.get("frameworks") or tuple(frameworks.FRAMEWORKS)
    repetitions = args.repetitions or preset.get("repetitions") or 1

    specs_to_run = specs.get_specs(circuits)
    runner_options = frameworks.RunnerOptions(
        dtype=args.dtype,
        mps_max_bond=args.mps_max_bond,
        legacy_site_packages=args.legacy_site_packages,
        legacy_auto_install=args.legacy_auto_install,
        legacy_version=args.legacy_version,
        legacy_cache_dir=args.legacy_cache_dir,
    ).to_dict()

    results: List[Dict[str, object]] = []
    for spec_obj in specs_to_run:
        for framework_name in frameworks_to_run:
            for run_id in range(repetitions):
                print(
                    f"Running {framework_name} on {spec_obj.name} (run {run_id + 1}/{repetitions})...",
                    flush=True,
                )
                payload = _execute_benchmark(spec_obj, framework_name, runner_options, run_id)
                results.append(payload)
                print(
                    f" -> status={payload['status']} runtime={payload['runtime_s']:.4f}s memory={payload['peak_memory_kb']:.0f}kB",
                    flush=True,
                )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "benchmark_results.csv"
    summary_csv = output_dir / "benchmark_summary.csv"

    _write_csv(detailed_csv, results)
    summary = _summarise(results)
    _write_csv(summary_csv, summary)

    if not args.no_plots:
        try:
            plots.create_summary_plots(summary, output_dir, metrics=["avg_runtime_s", "avg_peak_memory_kb"])
        except RuntimeError as exc:
            print(f"Plot generation skipped: {exc}")

    print(f"Detailed results written to {detailed_csv}")
    print(f"Summary written to {summary_csv}")

    failures = [row for row in results if row.get("status") != "ok"]
    if failures:
        print("Some benchmarks failed:")
        for fail in failures:
            print(f" - {fail['framework']} on {fail['circuit']}: {fail['details']}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
