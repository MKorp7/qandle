from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Gate:
    name: str
    wires: Tuple[int, ...]
    param_index: Optional[int] = None
