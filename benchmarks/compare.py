from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_metrics(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_rows(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def group_rows(rows: Sequence[Dict[str, str]], columns: Sequence[str]):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, "") for col in columns)
        grouped[key].append(row)
    return grouped


def compute_comparisons(
    groups,
    baseline_key: str | None,
    group_columns: Sequence[str],
) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, ...], Dict[str, str]]]:
    output_rows: List[Dict[str, str]] = []
    fastest_lookup: Dict[Tuple[str, ...], Dict[str, str]] = {}

    for group_key, rows in groups.items():
        baseline_rows = [row for row in rows if row.get("device_key") == baseline_key] if baseline_key else []
        if not baseline_rows and rows:
            baseline_rows = [rows[0]]

        baseline_by_metric: Dict[str, Dict[str, str]] = {}
        for base_row in baseline_rows:
            metric_name = base_row.get("metric_name", "")
            baseline_by_metric[metric_name] = base_row

        fastest_row = None
        fastest_time = math.inf

        for row in rows:
            metric_name = row.get("metric_name", "")
            base_row = baseline_by_metric.get(metric_name) or baseline_rows[0]

            time_s = to_float(row.get("execution_time_s"))
            base_time_s = to_float(base_row.get("execution_time_s"))
            speedup = base_time_s / time_s if time_s > 0 else float("nan")

            peak_cpu = to_float(row.get("peak_cpu_mb"))
            base_cpu = to_float(base_row.get("peak_cpu_mb"))
            delta_cpu = peak_cpu - base_cpu

            peak_gpu = to_float(row.get("peak_gpu_mb"))
            base_gpu = to_float(base_row.get("peak_gpu_mb"))
            delta_gpu = peak_gpu - base_gpu

            metric_value = row.get("metric_value", "")
            baseline_metric_value = base_row.get("metric_value", "")
            try:
                accuracy_delta = float(metric_value) - float(baseline_metric_value)
            except Exception:
                accuracy_delta = float("nan")

            result_row = {col: row.get(col, "") for col in group_columns}
            result_row.update(
                {
                    "device_key": row.get("device_key", ""),
                    "label": row.get("label", ""),
                    "hardware": row.get("hardware", ""),
                    "dtype": row.get("dtype", ""),
                    "time_s": f"{time_s:.6f}",
                    "time_std_s": row.get("execution_time_std_s", ""),
                    "speedup_vs_baseline": f"{speedup:.4f}" if not math.isnan(speedup) else "nan",
                    "peak_cpu_mb": f"{peak_cpu:.3f}",
                    "delta_cpu_mb": f"{delta_cpu:.3f}",
                    "peak_gpu_mb": f"{peak_gpu:.3f}",
                    "delta_gpu_mb": f"{delta_gpu:.3f}",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "accuracy_delta": f"{accuracy_delta:.6f}" if not math.isnan(accuracy_delta) else "nan",
                    "success": row.get("success", ""),
                    "error": row.get("error", ""),
                }
            )
            output_rows.append(result_row)

            if time_s > 0 and time_s < fastest_time:
                fastest_time = time_s
                fastest_row = result_row

        if fastest_row is not None:
            fastest_lookup[group_key] = fastest_row

    return output_rows, fastest_lookup


def print_summary(group_columns: Sequence[str], fastest_rows: Dict[Tuple[str, ...], Dict[str, str]]) -> None:
    print("=== Comparison Summary ===")
    for group_key, row in sorted(fastest_rows.items()):
        group_desc = ", ".join(f"{col}={value}" for col, value in zip(group_columns, group_key))
        label = row.get("label", row.get("device_key", ""))
        hardware = row.get("hardware", "")
        time_s = row.get("time_s", "")
        speedup = row.get("speedup_vs_baseline", "")
        print(f"Group [{group_desc}] -> fastest: {label} ({hardware}) time={time_s}s speedup={speedup}x")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results against a baseline device")
    parser.add_argument("--input", required=True, help="Raw benchmark CSV")
    parser.add_argument("--group-by", required=True, help="Comma separated column names to group by")
    parser.add_argument("--baseline", default=None, help="Baseline device key")
    parser.add_argument("--metrics", default="execution_time_s,peak_cpu_mb,peak_gpu_mb", help="Metrics to include in summary")
    parser.add_argument("--out", required=True, help="Output CSV path for comparisons")

    args = parser.parse_args(list(argv) if argv is not None else None)
    group_columns = parse_metrics(args.group_by)

    rows = load_rows(args.input)
    grouped = group_rows(rows, group_columns)

    comparisons, fastest_rows = compute_comparisons(grouped, args.baseline, group_columns)

    fieldnames = list(group_columns) + [
        "device_key",
        "label",
        "hardware",
        "dtype",
        "time_s",
        "time_std_s",
        "speedup_vs_baseline",
        "peak_cpu_mb",
        "delta_cpu_mb",
        "peak_gpu_mb",
        "delta_gpu_mb",
        "metric_name",
        "metric_value",
        "accuracy_delta",
        "success",
        "error",
    ]
    write_rows(args.out, fieldnames, comparisons)

    print_summary(group_columns, fastest_rows)


if __name__ == "__main__":
    main()

