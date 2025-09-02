"""Simple mapping utilities for parameter remapping.

This module previously exposed a function named :func:`none`; it has been
renamed to :func:`identity` to avoid confusion with Python's ``None``.  An
alias is kept for backward compatibility.
"""

import torch


def tanh(x):
    """Hyperbolic tangent mapping."""

    return torch.tanh(x)


def identity(x):
    """Return the input unchanged."""

    return x


# Backwards compatibility -----------------------------------------------------
none = identity
