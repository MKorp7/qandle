# ruff: noqa: F403 F401 F405

from typing import Any

from .qcircuit import *
from .measurements import *
from .embeddings import *
from .ansaetze import *
from .drawer import *
from .splitter import *
from . import config
from .convolution import *
from .errors import *
from .operators import *
from .noise import *
from .simulators import *
from .utils import *
from .utils import (
    do_not_implement,
    get_matrix_transforms,
    marginal_probabilities,
    parse_rot,
    reduce_dot,
)
from .utils.compile import compile, is_available as compile_available
from .qasm import *

from .utils_gates import H, X, Y, Z, S, T

__all__ = [
    "Circuit",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "compile",
    "compile_available",
    "config",
    "do_not_implement",
    "draw",
    "get_matrix_transforms",
    "marginal_probabilities",
    "parse_rot",
    "reduce_dot",
]


def __reimport() -> None:  # pragma: no cover
    print("reimporting qandle")
    import importlib
    import sys

    modules = {k: v for k, v in sys.modules.items()}
    for module in modules:
        if module.startswith("qandle"):
            importlib.reload(sys.modules[module])

    try:
        # Patch snakeviz to not show in notebook (always open in new tab)
        import snakeviz

        snakeviz.ipymagic._check_ipynb = lambda: False
    except ImportError:
        pass


def __count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)