"""Utility remapping functions used by qandle.

"""
from __future__ import annotations

from typing import Any

import torch


def _apply(element: Any, op):
    """Apply ``op`` to ``element`` while preserving the input type."""

    if torch.is_tensor(element):
        return op(element)
    if isinstance(element, (int, float)):
        return op(torch.tensor(element, dtype=torch.get_default_dtype())).item()
    return op(element)


def identity(x: Any) -> Any:
    """Return ``x`` unchanged."""

    return x


def none(x: Any) -> Any:
    """Alias of :func:`identity` for compatibility."""

    return x


def tanh(x: Any) -> Any:
    """Apply hyperbolic tangent element-wise."""

    return _apply(x, torch.tanh)
