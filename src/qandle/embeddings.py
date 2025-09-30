from __future__ import annotations

import abc
import copy
import itertools
import math
import random
import typing

import torch
import torch.nn.functional as F

from qandle.qcircuit import Circuit

import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm


class InputOperator(op.UnbuiltOperator, abc.ABC): ...


class InputOperatorBuilt(op.BuiltOperator, abc.ABC):
    @abc.abstractmethod
    def to_qasm(self) -> qasm.QasmRepresentation: ...

    def decompose(self):
        raise NotImplementedError(f"Decomposing {self.__class__} is not yet supported")

    def to_matrix(self):
        raise ValueError("Input operators do not have a matrix representation")


def _identity(input, *args, **kwargs):
    return input


def _broadcast_shape(
    shape_a: typing.Tuple[int, ...], shape_b: typing.Tuple[int, ...]
) -> typing.Tuple[int, ...]:
    """Return the broadcast shape for two shapes following PyTorch rules."""

    result: typing.List[int] = []
    for dim_a, dim_b in itertools.zip_longest(
        reversed(shape_a), reversed(shape_b), fillvalue=1
    ):
        if dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        elif dim_a == dim_b:
            result.append(dim_a)
        else:
            raise ValueError(
                f"Shapes {shape_a} and {shape_b} cannot be broadcast together."
            )
    return tuple(reversed(result))


def _shape_numel(shape: typing.Tuple[int, ...]) -> int:
    prod = 1
    for dim in shape:
        prod *= int(dim)
    return prod


class AmplitudeEmbeddingBuilt(InputOperatorBuilt):

    """
    Will ignore the state of the circuit and set the features as the state.
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        normalize: bool,
        pad_with: typing.Union[float, None],
        padding,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.normalize = normalize
        self.pad_with = pad_with
        self.padding = padding
        self.named = True
        self.name = name

    def forward(self, _=None, **kwargs):
        x = kwargs[self.name]
        pad_len = 2**self.num_qubits - x.shape[-1]
        x = self.padding(
            input=x,
            pad=(0, pad_len),
            mode="constant",
            value=self.pad_with,  # type: ignore
        )
        x = self.normalize(x, p=2, dim=-1)  # type: ignore
        x = torch.complex(real=x, imag=torch.zeros_like(x))
        return x

    def __str__(self) -> str:
        return "LegacyAmplitudeEmbedding"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("AmplitudeEmbedding is not yet supported")


class LegacyAmplitudeEmbedding(InputOperator):
    """
    Amplitude Embedding
    """

    def __init__(
        self,
        name: str,
        qubits: list,
        normalize: bool = False,
        pad_with: typing.Union[float, None] = None,
    ):
        self.name = name
        self.named = True
        self.qubits = qubits
        self.pad_with = pad_with
        if pad_with is not None:
            self.padding = torch.nn.functional.pad
        else:
            self.padding = _identity
        if normalize:
            self.normalize = torch.nn.functional.normalize
        else:
            self.normalize = _identity

    def build(self, num_qubits: int) -> LegacyAmplitudeEmbeddingBuilt:
        assert num_qubits == len(
            self.qubits
        ), "Current Implementation requires all qubits to be used."
        return LegacyAmplitudeEmbeddingBuilt(
            num_qubits=num_qubits,
            normalize=self.normalize,
            pad_with=self.pad_with,
            padding=self.padding,
            name=self.name,
        )

    def __str__(self) -> str:
        return "LegacyAmplitudeEmbedding"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("AmplitudeEmbedding is not yet supported")


class AngleEmbeddingBuilt(InputOperatorBuilt):
    """
    Will ignore the state of the circuit and set the features as the state.
    """

    def __init__(self, name: str, num_qubits: int, qubits: typing.List[int], rotation: str):
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits = qubits
        self.named = True
        self.name = name

        _0 = torch.tensor((0 + 0j,), requires_grad=False)
        _i = torch.tensor((0 + 1j,), requires_grad=False)
        self.register_buffer("_0", _0, persistent=False)
        self.register_buffer("_i", _i, persistent=False)
        self.rotation = rotation
        self.rots = utils.parse_rot(rotation)
        a, b, self.a_op, self.b_op = self._get_rot_matrices(self.rots, num_qubits, qubits)
        self.register_buffer("_a", a.contiguous(), persistent=False)
        self.register_buffer("_b", b.contiguous(), persistent=False)
        z = F.one_hot(
            torch.tensor(0, dtype=torch.long), num_classes=2**num_qubits
        ).to(torch.cfloat)
        self.register_buffer("_z", z, persistent=False)

    @staticmethod
    def _get_rot_matrices(rot, num_qubits, qubits):
        gates = [rot(qubit=w, remapping=None).build(num_qubits) for w in qubits]
        a = [g._a.T for g in gates]
        b = [g._b.T for g in gates]
        a = torch.stack(a, dim=0)
        b = torch.stack(b, dim=0)
        return a, b, gates[0].a_op, gates[0].b_op

    def forward(self, state=None, **kwargs):
        inp = kwargs[self.name]
        a_matrix = torch.einsum("dab,...d->...dab", self._a, self.a_op(inp / 2))
        b_matrix = torch.einsum("dab,...d->...dab", self._b, self.b_op(inp / 2))
        matrix = a_matrix + b_matrix
        prod = self.prod(matrix)
        return prod @ self._z

    def prod(self, inp):
        """
        Return the matrix product along the third-to-last dimension.
        Uses a vectorized approach to multiply all matrices in a single call
        to ``torch.linalg.multi_dot`` while supporting arbitrary batch
        dimensions. The previous implementation iterated with a Python loop,
        which was slow and could impede optimization.
        """
        *batch, k, n, m = inp.shape
        flat_inp = inp.reshape(-1, k, n, m)

        def _chain(mat_seq: torch.Tensor) -> torch.Tensor:
            return torch.linalg.multi_dot(mat_seq.unbind(0))

        res = torch.vmap(_chain)(flat_inp)
        return res.reshape(*batch, n, m)

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}").to_qasm())
        return reps  # type: ignore

    def __str__(self) -> str:
        return f"AngleEmbedding_{self.rotation}"

    def decompose(self):
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}"))
        return reps


class AngleEmbedding(InputOperator):
    """
    Angle Embedding
    """

    def __init__(
        self, name: str, qubits: typing.Union[typing.List[int], None] = None, rotation="rx"
    ):
        self.name = name
        self.named = True
        self.qubits = qubits
        self.rotation = rotation

    def build(self, num_qubits: int) -> AngleEmbeddingBuilt:
        if self.qubits is None:
            qubits = list(range(num_qubits))
        else:
            qubits = self.qubits
        return AngleEmbeddingBuilt(
            name=self.name, qubits=qubits, num_qubits=num_qubits, rotation=self.rotation
        )

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        r = hash(random.random())
        for w in self.qubits:
            reps.append(self.rots(qubit=w, name=f"angle_{r}_{w}").to_qasm())
        return reps  # type: ignore

    def __str__(self) -> str:
        return f"AngleEmbedding_{self.rotation}"


class _AngleEmbeddingBase(InputOperator):
    def __init__(
        self,
        n_qubits: int,
        *,
        name: str = "x",
        alpha: typing.Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
        reupload_layers: int = 0,
        projector: typing.Optional[
            typing.Union[
                torch.nn.Module,
                torch.Tensor,
                torch.nn.Parameter,
                typing.Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = None,
        qubits: typing.Optional[typing.Sequence[int]] = None,
        rotation: str,
    ):
        if reupload_layers < 0:
            raise ValueError("reupload_layers must be non-negative")
        if qubits is None:
            qubits = list(range(n_qubits))
        else:
            if len(qubits) != n_qubits:
                raise ValueError("Length of qubits must match n_qubits")
        if len(qubits) == 0:
            raise ValueError("At least one qubit must be provided for angle embedding")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices are not allowed")

        self.name = name
        self.named = True
        self.qubits = list(qubits)
        self._alpha = alpha
        self.reupload_layers = reupload_layers
        self.projector = projector
        self.rotation = rotation

    def build(self, num_qubits: int) -> "_AngleEmbeddingRotBuilt":
        max_qubit = max(self.qubits)
        if max_qubit >= num_qubits:
            raise ValueError(
                f"Embedding targets qubit {max_qubit}, but circuit has only {num_qubits} qubits"
            )

        projector = self.projector
        if isinstance(projector, torch.nn.Module):
            projector = copy.deepcopy(projector)

        return _AngleEmbeddingRotBuilt(
            name=self.name,
            num_qubits=num_qubits,
            qubits=self.qubits,
            rotation=self.rotation,
            alpha=self._alpha,
            reupload_layers=self.reupload_layers,
            projector=projector,
        )

    def __str__(self) -> str:
        return f"AngleEmbedding_{self.rotation}"


class _AngleEmbeddingRotBuilt(InputOperatorBuilt):
    def __init__(
        self,
        *,
        name: str,
        num_qubits: int,
        qubits: typing.Sequence[int],
        rotation: str,
        alpha: typing.Union[float, torch.Tensor, torch.nn.Parameter],
        reupload_layers: int,
        projector: typing.Optional[
            typing.Union[
                torch.nn.Module,
                torch.Tensor,
                torch.nn.Parameter,
                typing.Callable[[torch.Tensor], torch.Tensor],
            ]
        ],
    ):
        super().__init__()
        self.name = name
        self.named = True
        self.num_qubits = num_qubits
        self.qubits = list(qubits)
        self.num_embedding_qubits = len(self.qubits)
        self.num_layers = reupload_layers + 1
        if self.num_layers <= 0:
            raise ValueError("Number of embedding layers must be at least one")

        if isinstance(projector, torch.nn.Module):
            self.projector_module = projector
        else:
            self.projector_module = None

        self.projector_callable: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None
        if projector is not None and not isinstance(projector, torch.nn.Module):
            if isinstance(projector, torch.nn.Parameter):
                if projector.ndim != 2 or projector.shape[0] != self.num_embedding_qubits:
                    raise ValueError(
                        "Projector parameter must have shape (n_qubits, input_dim)"
                    )
                self.register_parameter("projector_param", projector)
            elif isinstance(projector, torch.Tensor):
                if projector.ndim != 2 or projector.shape[0] != self.num_embedding_qubits:
                    raise ValueError(
                        "Projector tensor must have shape (n_qubits, input_dim)"
                    )
                if projector.requires_grad:
                    self.register_parameter("projector_param", torch.nn.Parameter(projector))
                else:
                    self.register_buffer("projector_buffer", projector.clone().detach(), persistent=False)
            elif callable(projector):
                self.projector_callable = projector
            else:
                raise TypeError("Unsupported projector type")

        self.alpha_param: typing.Optional[torch.nn.Parameter] = None
        if isinstance(alpha, torch.nn.Parameter):
            self.alpha_param = alpha
            self.register_parameter("alpha_param", self.alpha_param)
        elif isinstance(alpha, torch.Tensor) and alpha.requires_grad:
            self.alpha_param = torch.nn.Parameter(alpha)
            self.register_parameter("alpha_param", self.alpha_param)
        else:
            self.register_buffer(
                "alpha_buffer",
                torch.as_tensor(alpha, dtype=torch.float32),
                persistent=False,
            )

        rot_cls = utils.parse_rot(rotation)
        a, b, self.a_op, self.b_op = AngleEmbeddingBuilt._get_rot_matrices(
            rot_cls, num_qubits, self.qubits
        )
        self.register_buffer("_a", a.contiguous(), persistent=False)
        self.register_buffer("_b", b.contiguous(), persistent=False)
        zero_state = F.one_hot(
            torch.tensor(0, dtype=torch.long), num_classes=2**num_qubits
        ).to(torch.cfloat)
        self.register_buffer("_zero_state", zero_state, persistent=False)

    def _alpha_tensor(
        self, batch_ndim: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if hasattr(self, "alpha_param") and self.alpha_param is not None:
            base = self.alpha_param.to(device=device, dtype=dtype)
        else:
            base = getattr(self, "alpha_buffer").to(device=device, dtype=dtype)

        if base.ndim == 0:
            alpha = base.reshape(1, 1).expand(self.num_layers, self.num_embedding_qubits)
        elif base.ndim == 1:
            if base.shape[0] == self.num_embedding_qubits:
                alpha = base.reshape(1, self.num_embedding_qubits).expand(
                    self.num_layers, self.num_embedding_qubits
                )
            elif base.shape[0] == self.num_layers:
                alpha = base.reshape(self.num_layers, 1).expand(
                    self.num_layers, self.num_embedding_qubits
                )
            else:
                raise ValueError(
                    "Alpha must be scalar, per-layer, per-qubit, or per-layer-per-qubit"
                )
        elif base.ndim == 2:
            if base.shape != (self.num_layers, self.num_embedding_qubits):
                raise ValueError(
                    "Alpha tensor must have shape (layers, n_qubits)"
                )
            alpha = base
        else:
            raise ValueError("Alpha tensor has unsupported number of dimensions")

        shape = (1,) * batch_ndim + (self.num_layers, self.num_embedding_qubits)
        return alpha.reshape(*shape)

    def _apply_projector(self, x: torch.Tensor) -> torch.Tensor:
        if self.projector_module is not None:
            self.projector_module = self.projector_module.to(device=x.device, dtype=x.dtype)
            original_shape = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            projected = self.projector_module(flat)
            return projected.reshape(*original_shape, projected.shape[-1])

        if hasattr(self, "projector_param"):
            proj = getattr(self, "projector_param").to(device=x.device, dtype=x.dtype)
            return torch.matmul(x, proj.transpose(-1, -2))

        if hasattr(self, "projector_buffer"):
            proj = getattr(self, "projector_buffer").to(device=x.device, dtype=x.dtype)
            return torch.matmul(x, proj.transpose(-1, -2))

        if self.projector_callable is not None:
            projected = self.projector_callable(x)
            if not isinstance(projected, torch.Tensor):
                projected = torch.as_tensor(projected, device=x.device, dtype=x.dtype)
            return projected

        return x

    def _match_qubit_count(self, x: torch.Tensor) -> torch.Tensor:
        last_dim = x.shape[-1]
        if last_dim > self.num_embedding_qubits:
            return x[..., : self.num_embedding_qubits]
        if last_dim < self.num_embedding_qubits:
            pad = self.num_embedding_qubits - last_dim
            zeros = torch.zeros(*x.shape[:-1], pad, device=x.device, dtype=x.dtype)
            return torch.cat([x, zeros], dim=-1)
        return x

    def _prepare_state(
        self, state: typing.Optional[torch.Tensor], device: torch.device
    ) -> typing.Tuple[torch.Tensor, typing.Tuple[int, ...]]:
        if state is None:
            zero = self._zero_state.to(device=device)
            return zero.reshape(1, -1), ()

        state_tensor = torch.as_tensor(state, device=device)
        if not state_tensor.dtype.is_complex:
            state_tensor = state_tensor.to(torch.cfloat)
        else:
            state_tensor = state_tensor.to(torch.cfloat)

        if state_tensor.shape[-1] != 2**self.num_qubits:
            raise ValueError(
                f"State dimension {state_tensor.shape[-1]} does not match circuit size"
            )

        batch_shape = tuple(int(dim) for dim in state_tensor.shape[:-1])
        return state_tensor.reshape(-1, state_tensor.shape[-1]), batch_shape

    def prod(self, inp: torch.Tensor) -> torch.Tensor:
        *batch, k, n, m = inp.shape
        flat_inp = inp.reshape(-1, k, n, m)

        def _chain(mat_seq: torch.Tensor) -> torch.Tensor:
            return torch.linalg.multi_dot(mat_seq.unbind(0))

        res = torch.vmap(_chain)(flat_inp)
        return res.reshape(*batch, n, m)

    def forward(self, state=None, **kwargs):
        if self.name not in kwargs:
            raise KeyError(f"Input key '{self.name}' not provided for angle embedding")

        features = kwargs[self.name]
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        if not features.dtype.is_floating_point:
            features = features.to(torch.get_default_dtype())

        device = features.device
        if isinstance(state, torch.Tensor):
            device = state.device
            features = features.to(device=device)

        features = self._apply_projector(features)
        features = features.to(device=device)
        features = self._match_qubit_count(features)

        feature_batch_shape = tuple(int(dim) for dim in features.shape[:-1])
        batch_ndim = len(feature_batch_shape)
        alpha = self._alpha_tensor(batch_ndim, device=device, dtype=features.dtype)
        angles = features.unsqueeze(-2) * alpha

        feature_batch_size = _shape_numel(feature_batch_shape)
        angles_flat = angles.reshape(feature_batch_size, self.num_layers, self.num_embedding_qubits)

        state_flat, state_batch_shape = self._prepare_state(state, device)
        state_batch_size = _shape_numel(state_batch_shape)

        final_batch_shape = _broadcast_shape(feature_batch_shape, state_batch_shape)
        final_batch_size = _shape_numel(final_batch_shape)

        if feature_batch_size == 1 and final_batch_size > 1:
            angles_flat = angles_flat.expand(final_batch_size, -1, -1)
        elif feature_batch_size != final_batch_size:
            raise ValueError(
                "Feature batch shape is incompatible with state batch shape for broadcasting"
            )

        if state_batch_size == 1 and final_batch_size > 1:
            state_flat = state_flat.expand(final_batch_size, -1)
        elif state_batch_size != final_batch_size:
            raise ValueError(
                "State batch shape is incompatible with feature batch shape for broadcasting"
            )

        a = self._a.to(device=device)
        b = self._b.to(device=device)
        state_flat = state_flat.to(device=device, dtype=torch.cfloat)

        for layer_idx in range(self.num_layers):
            layer_angles = angles_flat[:, layer_idx, :].to(device=device)
            half_angles = layer_angles / 2
            coeff_a = self.a_op(half_angles).to(device=device, dtype=a.dtype)
            coeff_b = self.b_op(half_angles).to(device=device, dtype=b.dtype)
            a_matrix = torch.einsum("qab,bq->bqab", a, coeff_a)
            b_matrix = torch.einsum("qab,bq->bqab", b, coeff_b)
            layer_matrix = a_matrix + b_matrix
            combined = self.prod(layer_matrix)
            state_flat = torch.einsum("bij,bj->bi", combined, state_flat)

        output = state_flat.reshape(*final_batch_shape, state_flat.shape[-1])
        if len(final_batch_shape) == 0:
            output = output.reshape(state_flat.shape[-1])
        return output


class AngleEmbeddingRX(_AngleEmbeddingBase):

    def __init__(
        self,
        n_qubits: int,
        *,
        name: str = "x",
        alpha: typing.Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
        reupload_layers: int = 0,
        projector: typing.Optional[
            typing.Union[
                torch.nn.Module,
                torch.Tensor,
                torch.nn.Parameter,
                typing.Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = None,
        qubits: typing.Optional[typing.Sequence[int]] = None,
    ):
        super().__init__(
            n_qubits,
            name=name,
            alpha=alpha,
            reupload_layers=reupload_layers,
            projector=projector,
            qubits=qubits,
            rotation="rx",
        )


class AngleEmbeddingRY(_AngleEmbeddingBase):

    def __init__(
        self,
        n_qubits: int,
        *,
        name: str = "x",
        alpha: typing.Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
        reupload_layers: int = 0,
        projector: typing.Optional[
            typing.Union[
                torch.nn.Module,
                torch.Tensor,
                torch.nn.Parameter,
                typing.Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = None,
        qubits: typing.Optional[typing.Sequence[int]] = None,
    ):
        super().__init__(
            n_qubits,
            name=name,
            alpha=alpha,
            reupload_layers=reupload_layers,
            projector=projector,
            qubits=qubits,
            rotation="ry",
        )


class AngleEmbeddingRZ(_AngleEmbeddingBase):

    def __init__(
        self,
        n_qubits: int,
        *,
        name: str = "x",
        alpha: typing.Union[float, torch.Tensor, torch.nn.Parameter] = 1.0,
        reupload_layers: int = 0,
        projector: typing.Optional[
            typing.Union[
                torch.nn.Module,
                torch.Tensor,
                torch.nn.Parameter,
                typing.Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = None,
        qubits: typing.Optional[typing.Sequence[int]] = None,
    ):
        super().__init__(
            n_qubits,
            name=name,
            alpha=alpha,
            reupload_layers=reupload_layers,
            projector=projector,
            qubits=qubits,
            rotation="rz",
        )
