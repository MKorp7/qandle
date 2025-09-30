"""Unified backend registry for benchmark suite.

This module exposes a lightweight abstraction layer that allows the
benchmark scripts in :mod:`benchmarks` to instantiate a wide range of
simulation backends using a single configuration interface.  Each backend
is described by :class:`BackendSpec`, which captures metadata, supported
features, default configuration values, and a factory used to build a
runtime handle.

The registry focuses on two families of backends that are commonly used in
internal benchmarking runs:

* QANDLE — Statevector and MPS simulators targeting CPU and GPU hardware.
* PennyLane — "default" reference simulators as well as lightning devices.

The concrete backend implementations might not always be available in the
execution environment (e.g., PennyLane or CUDA may be missing).  The
registry therefore performs lazy availability checks and gracefully marks
unavailable devices.  Benchmark scripts can call
``available_backends()`` to list the keys that can be used on the current
machine.  Attempting to instantiate an unavailable backend raises a
``RuntimeError`` with a descriptive message so that callers can surface the
problem to the user.

The registry intentionally keeps the returned ``handle`` objects opaque to
the benchmark layer.  Scripts only rely on metadata recorded in
``BackendInstance`` to document the configuration used for a benchmark run.
If an actual backend handle cannot be instantiated (for example when a
package is missing), ``handle`` is set to ``None`` and execution will fall
back to the reference NumPy simulator implemented in ``benchmarks.utils``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


AvailabilityChecker = Callable[[], Tuple[bool, Optional[str]]]
Factory = Callable[[int, int, Dict[str, Any]], Any]


def _load_module(name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Best-effort import helper.

    Returns ``(available, module, error_message)``.  The helper avoids
    importing the module twice when possible by inspecting ``sys.modules``
    first.  When the module cannot be imported ``module`` is ``None`` and
    ``error_message`` contains the exception string.
    """

    if name in globals():
        module = globals()[name]
        return True, module, None

    try:
        module = importlib.import_module(name)
        globals()[name] = module
        return True, module, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, None, str(exc)


def _torch_info() -> Tuple[bool, Optional[Any], Optional[str]]:
    available, torch_mod, err = _load_module("torch")
    if not available:
        return False, None, err
    try:
        cuda_available = bool(torch_mod.cuda.is_available())
    except Exception:  # pragma: no cover - fallback for odd torch builds
        cuda_available = False
    torch_mod.cuda_available = cuda_available  # type: ignore[attr-defined]
    return True, torch_mod, None


@dataclass
class BackendSpec:
    """Metadata and instantiation helpers for a backend."""

    key: str
    label: str
    family: str
    sim_type: str
    hardware: str
    features: Dict[str, Any]
    defaults: Dict[str, Any]
    factory: Factory
    availability: AvailabilityChecker = lambda: (True, None)
    _cached_availability: Optional[Tuple[bool, Optional[str]]] = field(
        default=None, init=False, repr=False
    )

    def is_available(self) -> bool:
        """Return ``True`` if the backend can be instantiated."""

        if self._cached_availability is None:
            self._cached_availability = self.availability()
        return bool(self._cached_availability[0])

    @property
    def unavailable_reason(self) -> Optional[str]:
        if self._cached_availability is None:
            self.is_available()
        if self._cached_availability:
            return self._cached_availability[1]
        return None

    def make(self, n_qubits: int, shots: int, **cfg: Any) -> "BackendInstance":
        """Instantiate the backend for ``n_qubits`` wires.

        ``cfg`` overrides the defaults declared for the backend.  Unsupported
        keys are ignored (after logging a warning) so that callers do not have
        to perform capability checks themselves.
        """

        if not self.is_available():
            reason = self.unavailable_reason or "backend is unavailable"
            raise RuntimeError(f"Backend '{self.key}' is not available: {reason}")

        applied = dict(self.defaults)
        applied.update({k: v for k, v in cfg.items() if v is not None})

        supported_dtypes = self.features.get("supports_dtype")
        requested_dtype = applied.get("dtype")
        if supported_dtypes and requested_dtype not in supported_dtypes:
            # Fall back to default dtype when the user requested an unsupported
            # value.  We avoid importing ``warnings`` at module import time to
            # keep the module lightweight in CLI contexts.
            import warnings

            warnings.warn(
                (
                    f"dtype '{requested_dtype}' is not supported by backend "
                    f"{self.key}. Falling back to '{self.defaults.get('dtype')}'."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            applied["dtype"] = self.defaults.get("dtype")

        if (
            applied.get("threads") is not None
            and not self.features.get("supports_threads", False)
        ):
            import warnings

            warnings.warn(
                f"Backend {self.key} does not support thread overrides; ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
            applied["threads"] = None

        if (
            applied.get("bond_dim") is not None
            and "bond_dim" not in self.features.get("knobs", [])
        ):
            import warnings

            warnings.warn(
                f"Backend {self.key} does not expose a bond_dim knob; ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
            applied["bond_dim"] = self.defaults.get("bond_dim")

        if (
            applied.get("splitting") is not None
            and "splitting" not in self.features.get("knobs", [])
        ):
            import warnings

            warnings.warn(
                f"Backend {self.key} does not expose a splitting knob; ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
            applied["splitting"] = self.defaults.get("splitting")

        handle = self.factory(n_qubits, shots, applied)
        return BackendInstance(spec=self, n_qubits=n_qubits, shots=shots, config=applied, handle=handle)


@dataclass
class BackendInstance:
    """Container returned by :meth:`BackendSpec.make`."""

    spec: BackendSpec
    n_qubits: int
    shots: int
    config: Dict[str, Any]
    handle: Any

    @property
    def dtype(self) -> Optional[str]:
        value = self.config.get("dtype")
        return str(value) if value is not None else None

    @property
    def threads(self) -> Optional[int]:
        return self.config.get("threads")

    @property
    def bond_dim(self) -> Optional[int]:
        return self.config.get("bond_dim")

    @property
    def splitting(self) -> Optional[bool]:
        return self.config.get("splitting")


_REGISTRY: Dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    _REGISTRY[spec.key] = spec


def get_backend(key: str) -> BackendSpec:
    try:
        return _REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown backend key '{key}'") from exc


def iter_backends(keys: Iterable[str]) -> Iterable[BackendSpec]:
    for key in keys:
        yield get_backend(key)


def available_backends() -> List[str]:
    return [key for key, spec in _REGISTRY.items() if spec.is_available()]


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def _qandle_factory(backend_name: str, device: str) -> Factory:
    def _factory(n_qubits: int, shots: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {
            "backend": backend_name,
            "device": device,
            "n_qubits": n_qubits,
            "shots": shots,
        }
        kwargs.update(cfg)
        return kwargs

    return _factory


def _check_qandle(new_backend: bool = True) -> Tuple[bool, Optional[str]]:
    env_var = "QANDLE_NEW_PATH" if new_backend else "QANDLE_PATH"
    qandle_path = os.environ.get(env_var)
    if not qandle_path:
        return False, f"environment variable {env_var} is not set"
    spec_name = "qandle" if new_backend else "qandle_old"
    module_available, _, err = _load_module(spec_name)
    if not module_available:
        return False, err or f"unable to import {spec_name}"
    return True, None


def _make_pennylane_device(name: str) -> Factory:
    def _factory(n_qubits: int, shots: int, cfg: Dict[str, Any]) -> Optional[Any]:
        available, qml, _ = _load_module("pennylane")
        if not available:
            return None

        device_kwargs: Dict[str, Any] = {"wires": n_qubits}
        device_dtype = cfg.get("dtype")
        if device_dtype is not None:
            device_kwargs["dtype"] = device_dtype
        if shots:
            device_kwargs["shots"] = shots
        try:
            device = qml.device(name, **device_kwargs)
        except Exception:
            return None
        return device

    return _factory


def _check_pennylane(device_name: str, require_cuda: bool = False) -> Tuple[bool, Optional[str]]:
    available, qml, err = _load_module("pennylane")
    if not available:
        return False, err
    if require_cuda:
        torch_available, torch_mod, torch_err = _torch_info()
        if not torch_available or not getattr(torch_mod, "cuda_available", False):
            return False, torch_err or "CUDA is not available"
    try:
        qml.device(device_name, wires=1)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _torch_cuda_available() -> bool:
    torch_available, torch_mod, _ = _torch_info()
    if not torch_available:
        return False
    return bool(getattr(torch_mod, "cuda_available", False))


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


register_backend(
    BackendSpec(
        key="qandle_sv_cpu",
        label="QANDLE (Statevector, CPU)",
        family="qandle",
        sim_type="statevector",
        hardware="CPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float32", "float64", "complex64", "complex128"],
            "supports_threads": True,
            "knobs": ["splitting"],
        },
        defaults={
            "dtype": "complex64",
            "threads": None,
            "bond_dim": None,
            "splitting": False,
        },
        factory=_qandle_factory("statevector", "cpu"),
        availability=lambda: _check_qandle(new_backend=True),
    )
)


register_backend(
    BackendSpec(
        key="qandle_sv_gpu",
        label="QANDLE (Statevector, GPU)",
        family="qandle",
        sim_type="statevector",
        hardware="GPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float32", "float64", "complex64", "complex128"],
            "supports_threads": False,
            "knobs": ["splitting"],
        },
        defaults={
            "dtype": "complex64",
            "threads": None,
            "bond_dim": None,
            "splitting": False,
        },
        factory=_qandle_factory("statevector", "cuda"),
        availability=lambda: (False, "CUDA backend unavailable")
        if not _torch_cuda_available()
        else _check_qandle(new_backend=True),
    )
)


register_backend(
    BackendSpec(
        key="qandle_mps_cpu",
        label="QANDLE (MPS, CPU)",
        family="qandle",
        sim_type="mps",
        hardware="CPU",
        features={
            "supports_grad": False,
            "supports_shots": True,
            "supports_dtype": ["float32", "float64", "complex64", "complex128"],
            "supports_threads": True,
            "knobs": ["bond_dim", "splitting"],
        },
        defaults={
            "dtype": "complex64",
            "threads": None,
            "bond_dim": 64,
            "splitting": False,
        },
        factory=_qandle_factory("mps", "cpu"),
        availability=lambda: _check_qandle(new_backend=True),
    )
)


register_backend(
    BackendSpec(
        key="pl_default_qubit",
        label="PennyLane default.qubit",
        family="pennylane",
        sim_type="statevector",
        hardware="CPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float32", "float64", "complex64", "complex128"],
            "supports_threads": False,
            "knobs": [],
        },
        defaults={
            "dtype": "float64",
            "threads": None,
            "bond_dim": None,
            "splitting": False,
        },
        factory=_make_pennylane_device("default.qubit"),
        availability=lambda: _check_pennylane("default.qubit"),
    )
)


register_backend(
    BackendSpec(
        key="pl_lightning_qubit",
        label="PennyLane lightning.qubit",
        family="pennylane",
        sim_type="statevector",
        hardware="CPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float64", "complex128"],
            "supports_threads": True,
            "knobs": [],
        },
        defaults={
            "dtype": "float64",
            "threads": None,
            "bond_dim": None,
            "splitting": False,
        },
        factory=_make_pennylane_device("lightning.qubit"),
        availability=lambda: _check_pennylane("lightning.qubit"),
    )
)


register_backend(
    BackendSpec(
        key="pl_lightning_gpu",
        label="PennyLane lightning.gpu",
        family="pennylane",
        sim_type="statevector",
        hardware="GPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float64", "complex128"],
            "supports_threads": False,
            "knobs": [],
        },
        defaults={
            "dtype": "float64",
            "threads": None,
            "bond_dim": None,
            "splitting": False,
        },
        factory=_make_pennylane_device("lightning.gpu"),
        availability=lambda: _check_pennylane("lightning.gpu", require_cuda=True),
    )
)


register_backend(
    BackendSpec(
        key="pl_default_tensor",
        label="PennyLane default.tensor",
        family="pennylane",
        sim_type="tensor",
        hardware="CPU",
        features={
            "supports_grad": True,
            "supports_shots": True,
            "supports_dtype": ["float64", "complex128"],
            "supports_threads": False,
            "knobs": ["bond_dim"],
        },
        defaults={
            "dtype": "float64",
            "threads": None,
            "bond_dim": 64,
            "splitting": False,
        },
        factory=_make_pennylane_device("default.tensor"),
        availability=lambda: _check_pennylane("default.tensor"),
    )
)


def list_backend_specs() -> List[BackendSpec]:
    return list(_REGISTRY.values())


__all__ = [
    "BackendInstance",
    "BackendSpec",
    "available_backends",
    "get_backend",
    "iter_backends",
    "list_backend_specs",
]

