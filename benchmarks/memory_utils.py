from __future__ import annotations

import os
import resource
from typing import Optional

import psutil


def _ru_maxrss_mb() -> Optional[float]:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return None
    # On Linux ru_maxrss is in kilobytes, on macOS it's bytes.
    value = usage.ru_maxrss
    if value == 0:
        return None
    if os.uname().sysname == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def get_peak_rss_mb() -> float:
    value = _ru_maxrss_mb()
    if value is not None:
        return float(value)
    process = psutil.Process()
    rss = process.memory_info().rss
    return float(rss) / (1024 * 1024)
