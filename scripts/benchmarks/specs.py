"""Circuit specifications used by the benchmarking suite."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

__all__ = [
    "CircuitSpec",
    "CIRCUIT_LIBRARY",
    "PRESETS",
    "get_specs",
    "resolve_preset",
    "generate_weights",
]


@dataclass(frozen=True)
class CircuitSpec:
    """Description of a benchmark circuit family."""

    name: str
    category: str
    num_qubits: int
    depth: int
    ansatz: str
    description: str
    seed: int = 7

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


CIRCUIT_LIBRARY: Dict[str, CircuitSpec] = {
    "small_verification": CircuitSpec(
        name="small_verification",
        category="verification",
        num_qubits=4,
        depth=4,
        ansatz="strongly_entangling",
        description="4-qubit SEL with moderate depth for regression-style verification",
        seed=13,
    ),
    "deep_narrow": CircuitSpec(
        name="deep_narrow",
        category="depth_stress",
        num_qubits=5,
        depth=32,
        ansatz="strongly_entangling",
        description="5-qubit SEL highlighting depth-heavy workloads",
        seed=23,
    ),
    "wide_low_entanglement": CircuitSpec(
        name="wide_low_entanglement",
        category="width_stress",
        num_qubits=12,
        depth=3,
        ansatz="two_local",
        description="12-qubit TwoLocal circuit emphasising low entanglement",
        seed=101,
    ),
}


PRESETS: Dict[str, Dict[str, object]] = {
    "full": {
        "circuits": tuple(CIRCUIT_LIBRARY),
        "frameworks": (
            "qandle-statevector",
            "qandle-mps",
            "pennylane",
            "qandle-legacy",
        ),
        "repetitions": 3,
    },
    "smoke": {
        "circuits": ("small_verification",),
        "frameworks": ("qandle-statevector",),
        "repetitions": 1,
    },
}


def get_specs(names: Sequence[str] | None = None) -> List[CircuitSpec]:
    """Return circuit specs for the given names."""

    if names is None:
        return list(CIRCUIT_LIBRARY.values())
    specs = []
    for name in names:
        try:
            specs.append(CIRCUIT_LIBRARY[name])
        except KeyError as exc:  # pragma: no cover - defensive branch
            available = ", ".join(sorted(CIRCUIT_LIBRARY))
            raise KeyError(f"Unknown circuit '{name}'. Available: {available}") from exc
    return specs


def resolve_preset(name: str | None) -> Dict[str, object]:
    """Resolve a preset description."""

    if name is None:
        return {}
    try:
        return PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Unknown preset '{name}'. Available presets: {', '.join(sorted(PRESETS))}.") from exc


def generate_weights(spec: CircuitSpec, dtype: str = "float32"):
    """Generate deterministic weights for a circuit specification."""

    import torch

    torch.manual_seed(spec.seed)
    try:
        torch_dtype = getattr(torch, dtype)
    except AttributeError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown torch dtype '{dtype}'.") from exc
    if spec.ansatz == "strongly_entangling":
        weights = torch.rand(spec.depth, spec.num_qubits, 3, dtype=torch_dtype)
    elif spec.ansatz == "two_local":
        weights = torch.rand(spec.depth, spec.num_qubits, dtype=torch_dtype)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported ansatz '{spec.ansatz}'.")
    return weights
