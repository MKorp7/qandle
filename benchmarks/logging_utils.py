from __future__ import annotations

import csv
import os
import socket
from datetime import datetime, timezone
from typing import Dict, Any


CSV_COLUMNS = [
    "timestamp_iso",
    "host",
    "framework",
    "algorithm",
    "n_qubits",
    "depth_or_p",
    "problem_id",
    "seed",
    "run_id",
    "execution_time_s",
    "peak_memory_mb",
    "accuracy_name",
    "accuracy_value",
    "success",
    "error",
]


def ensure_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    ensure_directory(csv_path)
    file_exists = os.path.exists(csv_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()

    row_with_meta = {**row, "timestamp_iso": timestamp, "host": host}

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row_with_meta.get(col, "") for col in CSV_COLUMNS})
