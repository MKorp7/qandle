"""Backend registry public API.

This package wraps the unified backend registry implementation in
:mod:`benchmarks.backends.registry` and re-exports the symbols that the
benchmark entry points depend on.  Keeping the registry inside the existing
package avoids clashes with the legacy backend modules that still live under
``benchmarks.backends``.
"""

from .registry import (
    available_backends,
    get_backend,
    iter_backends,
    list_backend_specs,
)

# ``BackendInstance`` is provided for compatibility with callers that relied on
# the old ``benchmarks.backends`` package re-exporting the container returned by
# ``BackendSpec.make``.  Importing it lazily keeps ``benchmarks.backends``
# importable even if the registry module is partially updated in a user
# worktree.  When the attribute is missing we fall back to a tiny placeholder so
# attribute access still succeeds, while the benchmarking entry points (which do
# not instantiate ``BackendInstance`` directly) continue to function.
try:  # pragma: no cover - defensive guard around partially synced trees
    from .registry import BackendInstance  # type: ignore
except ImportError:  # pragma: no cover
    class BackendInstance:  # type: ignore[empty-body]
        """Fallback placeholder used when the registry lacks ``BackendInstance``."""

        pass


__all__ = [
    "BackendInstance",
    "available_backends",
    "get_backend",
    "iter_backends",
    "list_backend_specs",
]
