from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import vmap
from torch.nn.utils.stateless import functional_call


"""
Parameter-shift gradients for circuits built from RX, RY and RZ gates.
"""

_SHIFT = math.pi / 2
_COFF = 0.5


@dataclass(frozen=True)
class _GradParamMeta:
    name: str
    shape: torch.Size
    size: int


def _eval_circuit(
    circuit: torch.nn.Module,
    state: torch.Tensor | None,
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Evaluate *circuit* with autograd disabled and return a detached scalar."""
    with torch.no_grad():
        out = circuit(state, **kwargs) if state is not None else circuit(**kwargs)
    if out.numel() != 1:
        raise RuntimeError(
            "Parameter-shift path expects the circuit to return a scalar "
            f"(e.g. an expectation value). Got shape {tuple(out.shape)}."
        )
    return out.squeeze().to(dtype=torch.float32)


def _collect_grad_param_names(circuit: torch.nn.Module) -> List[str]:
    return [name for name, param in circuit.named_parameters() if param.requires_grad]


def _build_base_param_dict(
    circuit: torch.nn.Module, device: torch.device
) -> OrderedDict[str, torch.Tensor]:
    base = OrderedDict()
    for name, param in circuit.named_parameters():
        base[name] = param.detach().to(device=device)
    return base


def _build_buffer_dict(
    circuit: torch.nn.Module, device: torch.device
) -> OrderedDict[str, torch.Tensor] | None:
    buffers = OrderedDict(
        (name, buf.detach().to(device=device)) for name, buf in circuit.named_buffers()
    )
    return buffers or None


def _grad_param_meta(
    names: Sequence[str], params: Sequence[torch.Tensor]
) -> List[_GradParamMeta]:
    return [
        _GradParamMeta(name=name, shape=param.shape, size=param.numel())
        for name, param in zip(names, params)
    ]


def _flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    flats = [p.reshape(-1) for p in params]
    return torch.cat(flats) if flats else torch.empty(0)


def _functional_eval(
    circuit: torch.nn.Module,
    base_params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor] | None,
    grad_meta: Sequence[_GradParamMeta],
    flat_values: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    params_dict = base_params.copy()
    offset = 0
    for meta in grad_meta:
        next_offset = offset + meta.size
        view = flat_values[offset:next_offset].view(meta.shape)
        params_dict[meta.name] = view
        offset = next_offset
    return functional_call(circuit, params_dict, args=args, kwargs=kwargs, buffers=buffers)


def _batched_parameter_shift(
    circuit: torch.nn.Module,
    state: torch.Tensor | None,
    kwargs: Dict[str, Any],
    params: Sequence[torch.Tensor],
    param_names: Sequence[str],
) -> torch.Tensor:
    if not params:
        return torch.empty(0, dtype=torch.float32)

    grad_meta = _grad_param_meta(param_names, params)
    flat_base = _flatten_params(params)
    num_params = flat_base.numel()
    if num_params == 0:
        return torch.tensor([], device=flat_base.device)

    device = flat_base.device
    dtype = flat_base.dtype
    eye = torch.eye(num_params, device=device, dtype=dtype)
    plus = flat_base.unsqueeze(0) + _SHIFT * eye
    minus = flat_base.unsqueeze(0) - _SHIFT * eye

    args: Tuple[Any, ...]
    if state is None:
        args = tuple()
    else:
        args = (state,)

    base_params = _build_base_param_dict(circuit, device=device)
    buffers = _build_buffer_dict(circuit, device=device)

    def _apply(flat_batch: torch.Tensor) -> torch.Tensor:
        return _functional_eval(
            circuit,
            base_params,
            buffers,
            grad_meta,
            flat_batch,
            args,
            kwargs,
        )

    f_plus = vmap(_apply)(plus)
    f_minus = vmap(_apply)(minus)
    return _COFF * (f_plus - f_minus)


class _ParameterShiftFn(torch.autograd.Function):
    """Autograd function implementing the parameter-shift rule."""

    @staticmethod
    def forward(ctx, *args):
        *params, circuit, state, kwargs = args
        ctx.circuit = circuit
        ctx.state = state
        ctx.kwargs = kwargs
        param_names = _collect_grad_param_names(circuit)
        if len(param_names) != len(params):
            raise RuntimeError(
                "Mismatch between saved parameters and circuit parameters requiring gradients."
            )
        ctx.param_names = param_names
        ctx.save_for_backward(*params)
        return _eval_circuit(circuit, state, kwargs)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_out = grad_output[0]
        params: Tuple[torch.Tensor, ...] = ctx.saved_tensors
        circuit, state, kwargs = ctx.circuit, ctx.state, ctx.kwargs
        param_names: Sequence[str] = ctx.param_names

        with torch.inference_mode():
            grad_values = _batched_parameter_shift(
                circuit,
                state,
                kwargs,
                params,
                param_names,
            )

        grads: List[torch.Tensor] = []
        offset = 0
        for param, meta in zip(params, _grad_param_meta(param_names, params)):
            next_offset = offset + meta.size
            grad_flat = grad_values[offset:next_offset]
            grad_tensor = grad_flat.view(param.shape)
            grads.append(grad_out * grad_tensor)
            offset = next_offset

        grads.extend([None, None, None])
        return tuple(grads)


def parameter_shift_forward(
    circuit: torch.nn.Module,
    state: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Execute *circuit* while enabling parameter-shift gradients."""
    params = [p for p in circuit.parameters() if p.requires_grad]
    return _ParameterShiftFn.apply(*params, circuit, state, kwargs)

