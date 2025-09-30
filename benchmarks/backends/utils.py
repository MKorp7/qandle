from __future__ import annotations

import math
from typing import Optional

import torch

from benchmarks.ir.gates import Gate


def get_gate_angle(gate: Gate, params: torch.Tensor) -> Optional[torch.Tensor]:
    if gate.param_index is None:
        return None
    return params[gate.param_index]


def rotation_matrix(axis: str, angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    half = angle / 2.0
    cos = torch.cos(half).to(dtype)
    sin = torch.sin(half).to(dtype)
    zero = torch.zeros((), dtype=dtype, device=angle.device)
    if axis == "RX":
        return torch.stack([
            torch.stack([cos, -1j * sin]),
            torch.stack([-1j * sin, cos]),
        ])
    if axis == "RY":
        return torch.stack([
            torch.stack([cos, -sin]),
            torch.stack([sin, cos]),
        ])
    if axis == "RZ":
        return torch.stack([
            torch.stack([torch.exp(-0.5j * angle), zero]),
            torch.stack([zero, torch.exp(0.5j * angle)]),
        ])
    raise ValueError(f"Unsupported rotation axis {axis}")


def apply_gate_matrix(backend, gate: Gate, params: torch.Tensor, dtype: torch.dtype) -> None:
    angle = get_gate_angle(gate, params)
    if gate.name in {"RX", "RY", "RZ"}:
        if angle is None:
            raise ValueError(f"Gate {gate.name} requires param_index")
        matrix = rotation_matrix(gate.name, angle, dtype)
        backend.apply_1q(matrix, gate.wires[0])
    elif gate.name == "H":
        matrix = torch.tensor(
            [[1, 1], [1, -1]], dtype=dtype, device=params.device
        ) / math.sqrt(2.0)
        backend.apply_1q(matrix, gate.wires[0])
    elif gate.name == "CNOT":
        matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=dtype,
            device=params.device,
        )
        backend.apply_2q(matrix, gate.wires[0], gate.wires[1])
    elif gate.name == "CZ":
        matrix = torch.diag(torch.tensor([1, 1, 1, -1], dtype=dtype, device=params.device))
        backend.apply_2q(matrix, gate.wires[0], gate.wires[1])
    else:
        raise ValueError(f"Unsupported gate {gate.name}")
