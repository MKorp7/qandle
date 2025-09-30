from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkloadResult:
    execution_time_s: float
    peak_memory_mb: float
    accuracy_name: str
    accuracy_value: float
    success: bool
    error: str | None = None
