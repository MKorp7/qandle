from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _load(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    benchmarks = payload.get("benchmarks", [])
    return {entry["fullname"]: entry for entry in benchmarks}


def _format_row(name: str, baseline: float, current: float) -> str:
    delta = current - baseline
    pct = 0.0 if baseline == 0 else delta / baseline * 100.0
    return f"{name}|{baseline:.6e}|{current:.6e}|{pct:+.2f}%"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pytest-benchmark JSON runs")
    parser.add_argument("baseline", type=Path, help="Path to the baseline JSON file")
    parser.add_argument("current", type=Path, help="Path to the current run JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Allowed relative regression before failing (default: 0.2, i.e. 20%)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write a Markdown table summary",
    )
    args = parser.parse_args(argv)

    baseline_data = _load(args.baseline)
    current_data = _load(args.current)

    missing = sorted(set(baseline_data) - set(current_data))
    if missing:
        raise SystemExit(f"Missing benchmarks in current run: {', '.join(missing)}")

    rows: list[str] = ["benchmark|baseline_mean|current_mean|delta"]
    regressions: list[str] = []

    for name, base_entry in sorted(baseline_data.items()):
        base_mean = float(base_entry["stats"]["mean"])
        curr_mean = float(current_data[name]["stats"]["mean"])
        if base_mean == 0.0:
            continue
        rel = (curr_mean - base_mean) / base_mean
        if rel > args.threshold:
            regressions.append(
                f"{name}: {rel * 100.0:+.2f}% slower (baseline {base_mean:.6e}s, current {curr_mean:.6e}s)"
            )
        rows.append(_format_row(name, base_mean, curr_mean))

    output = "\n".join(rows)
    print("Benchmark comparison (mean time):")
    print(output)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            handle.write("| benchmark | baseline_mean | current_mean | delta |\n")
            handle.write("|---|---|---|---|\n")
            for row in rows[1:]:
                bench, base, curr, delta = row.split("|")
                handle.write(f"| {bench} | {base} | {curr} | {delta} |\n")

    if regressions:
        print("\nDetected regressions:")
        for line in regressions:
            print(f" - {line}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
