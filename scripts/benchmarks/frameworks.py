"""Execution helpers for the benchmarking suite."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict

from . import specs

__all__ = [
    "FrameworkInfo",
    "FRAMEWORKS",
    "RunnerOptions",
    "run_framework_once",
]


@dataclass(frozen=True)
class FrameworkInfo:
    """Metadata about an execution backend."""

    name: str
    label: str
    description: str


@dataclass
class RunnerOptions:
    """Options passed to framework runners."""

    dtype: str = "float32"
    device: str = "cpu"
    mps_max_bond: int = 128
    legacy_version: str = "0.1.12"
    legacy_site_packages: str | None = None
    legacy_auto_install: bool = False
    legacy_cache_dir: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FRAMEWORKS: Dict[str, FrameworkInfo] = {
    "qandle-statevector": FrameworkInfo(
        name="qandle-statevector",
        label="Qandle (statevector)",
        description="Current Qandle statevector simulator",
    ),
    "qandle-mps": FrameworkInfo(
        name="qandle-mps",
        label="Qandle (MPS)",
        description="Current Qandle matrix-product-state backend",
    ),
    "pennylane": FrameworkInfo(
        name="pennylane",
        label="PennyLane",
        description="Reference PennyLane simulator (default.qubit.torch)",
    ),
    "qandle-legacy": FrameworkInfo(
        name="qandle-legacy",
        label="Qandle (legacy)",
        description="Released pip version of Qandle",
    ),
}


def _build_qandle_circuit(qandle_mod, spec: specs.CircuitSpec, weights):
    qubits = list(range(spec.num_qubits))
    if spec.ansatz == "strongly_entangling":
        ansatz = qandle_mod.StronglyEntanglingLayer(qubits=qubits, depth=spec.depth, q_params=weights)
    elif spec.ansatz == "two_local":
        ansatz = qandle_mod.TwoLocal(qubits=qubits, depth=spec.depth, q_params=weights)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported ansatz '{spec.ansatz}'.")
    built = ansatz.build(num_qubits=spec.num_qubits)
    return qandle_mod.Circuit([built], num_qubits=spec.num_qubits)


def _run_qandle_statevector(spec: specs.CircuitSpec, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyTorch is required for Qandle benchmarks. Install via 'pip install torch'.") from exc
    import qandle

    weights = specs.generate_weights(spec, dtype=options.get("dtype", "float32"))
    circuit = _build_qandle_circuit(qandle, spec, weights)
    state = circuit()
    norm = torch.linalg.vector_norm(state).item()
    return {"state_norm": norm}


def _run_qandle_mps(spec: specs.CircuitSpec, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyTorch is required for Qandle benchmarks. Install via 'pip install torch'.") from exc
    import qandle

    weights = specs.generate_weights(spec, dtype=options.get("dtype", "float32"))
    circuit = _build_qandle_circuit(qandle, spec, weights)
    backend_kwargs = {"max_bond_dim": int(options.get("mps_max_bond", 128))}
    backend = circuit.forward(backend="mps", backend_kwargs=backend_kwargs)
    probs = backend.measure()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
    return {"entropy_bits": entropy, "max_bond_used": getattr(backend, "max_bond_used", None)}


def _run_pennylane(spec: specs.CircuitSpec, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyTorch is required for PennyLane benchmarks.") from exc
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PennyLane is required. Install via 'pip install pennylane'.") from exc

    weights = specs.generate_weights(spec, dtype=options.get("dtype", "float32"))
    dev = qml.device("default.qubit.torch", wires=spec.num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit():
        if spec.ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights=weights, wires=range(spec.num_qubits))
        elif spec.ansatz == "two_local":
            qml.BasicEntanglerLayers(weights=weights, wires=range(spec.num_qubits))
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported ansatz '{spec.ansatz}'.")
        return qml.state()

    state = circuit()
    norm = torch.linalg.vector_norm(state).item()
    return {"state_norm": norm}


def _ensure_legacy_path(options: Dict[str, Any]) -> Path:
    if options.get("legacy_site_packages"):
        return Path(options["legacy_site_packages"]).expanduser().resolve()
    if not options.get("legacy_auto_install"):
        raise RuntimeError(
            "Legacy Qandle path not provided. Use --legacy-site-packages or enable --legacy-auto-install."
        )
    version = options.get("legacy_version", "0.1.12")
    cache_dir = Path(options.get("legacy_cache_dir") or Path.home() / ".cache" / "qandle-benchmarks")
    target = cache_dir / f"qandle-{version}"
    if not (target / "qandle").exists():
        target.mkdir(parents=True, exist_ok=True)
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "pip", "install", f"qandle=={version}", "--target", str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to install legacy Qandle. Provide an existing path via --legacy-site-packages.\n" + proc.stderr
            )
    return target


def _run_qandle_legacy(spec: specs.CircuitSpec, options: Dict[str, Any]) -> Dict[str, Any]:
    import sys
    from importlib import import_module

    legacy_path = _ensure_legacy_path(options)
    repo_root = Path(__file__).resolve().parents[2]
    sys_path = [p for p in sys.path if Path(p).resolve() != repo_root]
    sys_path.insert(0, str(legacy_path))
    sys.path[:] = sys_path
    legacy = import_module("qandle")
    weights = specs.generate_weights(spec, dtype=options.get("dtype", "float32"))
    circuit = _build_qandle_circuit(legacy, spec, weights)
    state = circuit()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyTorch is required for Qandle benchmarks. Install via 'pip install torch'.") from exc

    norm = torch.linalg.vector_norm(state).item()
    return {"state_norm": norm, "legacy_version": getattr(legacy, "__version__", None)}


_RUNNERS: Dict[str, Callable[[specs.CircuitSpec, Dict[str, Any]], Dict[str, Any]]] = {
    "qandle-statevector": _run_qandle_statevector,
    "qandle-mps": _run_qandle_mps,
    "pennylane": _run_pennylane,
    "qandle-legacy": _run_qandle_legacy,
}


def run_framework_once(framework: str, spec: specs.CircuitSpec, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        runner = _RUNNERS[framework]
    except KeyError as exc:
        available = ", ".join(sorted(_RUNNERS))
        raise KeyError(f"Unknown framework '{framework}'. Available: {available}") from exc
    return runner(spec, options)
