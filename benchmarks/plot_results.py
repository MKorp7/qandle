from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


try:  # Optional dependency
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency guard
    plt = None  # type: ignore
    _HAS_MPL = False


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def group_rows(rows: Sequence[Dict[str, str]], columns: Sequence[str]):
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, "") for col in columns)
        grouped[key].append(row)
    return grouped


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def plot_group(
    group_key: Tuple[str, ...],
    rows: Sequence[Dict[str, str]],
    group_columns: Sequence[str],
    x_axis: str,
    metric: str,
    output_prefix: str,
) -> None:
    if not _HAS_MPL:
        print("matplotlib is not installed; skipping plot generation")
        return

    data: Dict[str, List[Tuple[float, float, Dict[str, str]]]] = defaultdict(list)
    for row in rows:
        try:
            x_value = to_float(row.get(x_axis, ""))
            y_value = to_float(row.get(metric, ""))
        except Exception:
            continue
        if x_value != x_value or y_value != y_value:  # NaN guard
            continue
        device = row.get("device_key", "")
        label = row.get("label", device)
        hardware = row.get("hardware", "")
        dtype = row.get("dtype", "")
        data[label].append((x_value, y_value, {"hardware": hardware, "dtype": dtype}))

    if not data:
        print(f"No numeric data available for group {group_key}")
        return

    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "v", "D", "*", "x", "+"]
    for index, (label, points) in enumerate(sorted(data.items())):
        points_sorted = sorted(points, key=lambda item: item[0])
        xs = [item[0] for item in points_sorted]
        ys = [item[1] for item in points_sorted]
        marker = markers[index % len(markers)]
        info = points_sorted[0][2]
        pretty_label = f"{label} ({info['hardware']}, {info['dtype']})"
        plt.plot(xs, ys, marker=marker, label=pretty_label)

    title = ", ".join(f"{key}={value}" for key, value in zip(group_columns, group_key))
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = f"{output_prefix}_{'_'.join(str(v) for v in group_key)}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot -> {output_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results grouped by backend")
    parser.add_argument("--input", required=True, help="Benchmark CSV file")
    parser.add_argument("--output-prefix", default="plot", help="Prefix for generated plot files")
    parser.add_argument("--group-by", default="algorithm,scenario", help="Comma separated columns for grouping")
    parser.add_argument("--x-axis", default="n_qubits", help="Column to use as X axis")
    parser.add_argument("--metric", default="execution_time_s", help="Metric column to plot on Y axis")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not _HAS_MPL:
        print("matplotlib is not available; install it to enable plotting")
        return

    rows = load_rows(args.input)
    group_columns = [item.strip() for item in args.group_by.split(",") if item.strip()]
    groups = group_rows(rows, group_columns)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    for group_key, group_rows_list in groups.items():
        plot_group(group_key, group_rows_list, group_columns, args.x_axis, args.metric, args.output_prefix)


if __name__ == "__main__":
    main()

