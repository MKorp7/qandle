from __future__ import annotations

import importlib
import os
import sys
from typing import Protocol

from .ir.gates import Gate  # type: ignore  # for type checking only


class Backend(Protocol):
    name: str

    def simulate_state(self, n_qubits: int, gates: list[Gate], params, seed: int): ...


def import_pl_backend() -> Backend:
    module = importlib.import_module("benchmarks.backends.pl_backend")
    return module.PennyLaneBackend()


def import_qandle_origin_backend() -> Backend:
    module = importlib.import_module("benchmarks.backends.qandle_origin_backend")
    return module.QandleOriginBackend()


def import_qandle_new_backend(qandle_new_path: str) -> Backend:
    if not qandle_new_path:
        raise ValueError("QANDLE_NEW_PATH environment variable is required for qandle_new backend")
    if qandle_new_path not in sys.path:
        sys.path.insert(0, qandle_new_path)
    src_path = os.path.join(qandle_new_path, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
    os.environ.setdefault("QANDLE_NEW_PATH", qandle_new_path)
    module = importlib.import_module("benchmarks.backends.qandle_new_backend")
    return module.QandleNewBackend()
