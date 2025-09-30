"""Utilities for working with :func:`torch.compile`."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, cast

import torch

Compiler = Callable[[Callable[..., Any]], Callable[..., Any]]


@contextmanager
def compile(**kwargs: Any) -> Iterator[Compiler]:
    """Yield a ``torch.compile`` wrapper with graceful degradation."""

    if not hasattr(torch, "compile"):
        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        yield _identity
        return

    def _compiler(fn: Callable[..., Any]) -> Callable[..., Any]:
        compiled = torch.compile(fn, **kwargs)
        return cast(Callable[..., Any], compiled)

    yield _compiler


def is_available() -> bool:
    """Return ``True`` if :func:`torch.compile` is supported."""

    return hasattr(torch, "compile")


__all__ = ["compile", "is_available"]
