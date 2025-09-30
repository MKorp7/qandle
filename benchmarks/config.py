from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    algorithms: List[str] = field(default_factory=lambda: ["vqe_h2", "qaoa", "classifier"])
    frameworks: List[str] = field(default_factory=lambda: ["pennylane", "qadle_origin", "qandle_new"])
    n_list: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    depth_list: List[int] = field(default_factory=lambda: [1, 3, 5])
    seed: int = 123
    output_csv: str = "benchmarks/out/results.csv"
    parallel: bool = False
    max_workers: int | None = None
    quick_smoke: bool = False


DEFAULT_CONFIG = BenchmarkConfig()
