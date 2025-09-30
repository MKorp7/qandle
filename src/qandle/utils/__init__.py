from __future__ import annotations

from typing import Any, Callable, Sequence, cast

import einops.layers.torch as einl
import torch


def reduce_dot(*args: torch.Tensor) -> torch.Tensor:
    """Compute the dot product of a series of matrices."""

    if len(args) == 1:
        return args[0]
    matrix = args[0]
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    for sub_matrix in args[1:]:
        if sub_matrix.dim() == 2:
            sm = sub_matrix.unsqueeze(0)
        elif sub_matrix.dim() == 3:
            sm = sub_matrix
        else:
            raise ValueError(f"Invalid shape {sub_matrix.shape}")
        matrix = torch.matmul(sm, matrix)
    return matrix


def marginal_probabilities(
    probabilities: torch.Tensor,
    qubits: Sequence[int] | int | None,
    n_qubits: int,
) -> torch.Tensor:
    """Aggregate marginal probabilities for ``qubits``."""

    if qubits is None:
        return probabilities

    if isinstance(qubits, int):
        qubit_list: list[int] = [qubits]
    else:
        qubit_list = list(qubits)

    if len(qubit_list) == 0:
        result_dtype = probabilities.dtype
        return torch.ones(1, dtype=result_dtype, device=probabilities.device)

    indices = torch.arange(probabilities.shape[0], device=probabilities.device)
    shifts = torch.tensor(
        [n_qubits - q - 1 for q in qubit_list],
        device=probabilities.device,
        dtype=indices.dtype,
    )
    bits = ((indices.unsqueeze(-1) >> shifts) & 1).to(indices.dtype)
    weights = (1 << torch.arange(len(qubit_list), device=probabilities.device, dtype=indices.dtype))
    keys = (bits * weights).sum(dim=-1)
    out = torch.zeros(1 << len(qubit_list), dtype=probabilities.dtype, device=probabilities.device)
    return torch.scatter_add(out, 0, keys, probabilities)


def parse_rot(rot: str) -> type[torch.nn.Module]:
    from .. import operators as op

    rot = rot.lower().replace("r", "")
    if rot == "x":
        return cast(type[torch.nn.Module], op.RX)
    elif rot == "y":
        return cast(type[torch.nn.Module], op.RY)
    elif rot == "z":
        return cast(type[torch.nn.Module], op.RZ)
    else:
        raise ValueError(f"Unknown rotation {rot}")


def do_not_implement(*protected: str, reason: str = "") -> type:
    class LimitedClass(type):
        def __new__(
            cls,
            name: str,
            bases: tuple[type[Any], ...],
            attrs: dict[str, Any],
        ) -> type[Any]:
            for attribute in attrs:
                if attribute in protected:
                    if reason:
                        raise AttributeError(reason)
                    else:
                        raise AttributeError(f'Overiding of attribute "{attribute}" not allowed')
            return super().__new__(cls, name, bases, attrs)

    return LimitedClass


def get_matrix_transforms(
    num_qubits: int, in_circuit: Sequence[int]
) -> tuple[einl.Rearrange, einl.Rearrange]:
    """Get einops layers for transforming between state and matrix representations."""

    qubits_in_subc = [f"a{i}" for i in in_circuit]
    qubits_not_in_subc = [f"a{i}" for i in range(num_qubits) if i not in in_circuit]
    all_qubits = [f"a{i}" for i in range(num_qubits)]
    state_pattern = f"batch ({' '.join(all_qubits)})"
    matrix_pattern = f"(batch {' '.join(qubits_not_in_subc)}) ({' '.join(qubits_in_subc)})"
    to_matrix_pattern = f"{state_pattern} -> {matrix_pattern}"
    to_state_pattern = f"{matrix_pattern} -> {state_pattern}"
    args = {f"a{i}": 2 for i in range(num_qubits)}
    to_matrix = einl.Rearrange(to_matrix_pattern, **args)
    to_state = einl.Rearrange(to_state_pattern, **args)
    return to_matrix, to_state


def _default_loss(value: torch.Tensor) -> torch.Tensor:
    return value[0]


def __analyze_barren_plateu(
    circuit: torch.nn.Module,
    loss_f: Callable[[torch.Tensor], torch.Tensor] = _default_loss,
    num_points: int = 30,
    other_params: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:  # pragma: no cover
    """Analyze gradients of a circuit to detect barren plateaus."""

    possible_params = torch.linspace(-torch.pi, torch.pi, num_points, requires_grad=True)
    zerosd = {
        k: torch.ones_like(v, requires_grad=True) + other_params
        for k, v in circuit.state_dict().items()
    }
    grads = torch.zeros(len(zerosd), num_points)
    losses = torch.zeros(len(zerosd), num_points)
    for ipname in range(len(zerosd)):
        pname = list(zerosd.keys())[ipname]
        for thetai, theta in enumerate(possible_params):
            sd = zerosd.copy()
            sd[pname] = theta
            circuit.load_state_dict(sd)
            for param in circuit.parameters():
                param.requires_grad = True
            circuit.zero_grad()
            loss = loss_f(circuit())
            loss.backward()
            grad = list(circuit.parameters())[ipname].grad
            grads[ipname, thetai] = grad.abs().mean()
            losses[ipname, thetai] = loss.detach()
    return grads, losses, possible_params.detach(), list(circuit.state_dict().keys())
