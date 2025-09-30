"""Plotting helpers for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Mapping

__all__ = ["create_summary_plots"]


def _prepare_series(results: Iterable[Mapping[str, object]], metric: str):
    circuits = sorted({str(r["circuit"]) for r in results})
    frameworks = sorted({str(r["framework"]) for r in results})
    data = defaultdict(lambda: {c: float("nan") for c in circuits})
    for row in results:
        if row.get("status") != "ok":
            continue
        values = data[row["framework"]]
        values[row["circuit"]] = float(row[metric])
    return circuits, frameworks, data


def create_summary_plots(
    results: Iterable[Mapping[str, object]],
    output_dir: Path,
    metrics: List[str] | None = None,
) -> List[Path]:
    """Create bar plots for the requested metrics.

    Parameters
    ----------
    results:
        Iterable of result dictionaries as produced by ``run.py``.
    output_dir:
        Directory where the plots should be written.
    metrics:
        Names of metrics to plot. Defaults to runtime and peak memory.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to generate plots") from exc

    metrics = metrics or ["runtime_s", "peak_memory_kb"]
    created: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        circuits, frameworks, data = _prepare_series(results, metric)
        if not frameworks:
            continue
        width = 0.8 / max(len(frameworks), 1)
        fig, ax = plt.subplots(figsize=(max(6, len(circuits) * 1.5), 4))
        positions = range(len(circuits))
        for idx, framework in enumerate(frameworks):
            offsets = [p + idx * width - 0.5 + width * len(frameworks) / 2 for p in positions]
            values = [data[framework][c] for c in circuits]
            ax.bar(offsets, values, width=width, label=framework)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(circuits, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Benchmark comparison: {metric}")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / f"{metric}.png"
        fig.savefig(out_path)
        created.append(out_path)
        plt.close(fig)
    return created
