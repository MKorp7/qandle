import torch
import torch.nn.functional as F
from qandle import drawer
from qandle import splitter
from qandle import operators
from qandle import qasm
from qandle import utils
from qandle.kernels import apply_CCNOT, apply_CNOT
from qandle.backends import (
    DensityMatrixBackend,
    MPSBackend,
    PauliTransferMatrixBackend,
    QuantumBackend,
    StabilizerBackend,
    StateVectorBackend,
)
from qandle.noise.channels import BuiltNoiseChannel
import warnings
import typing

_BACKENDS = {
    "statevector": StateVectorBackend,
    "mps": MPSBackend,
    "density_matrix": DensityMatrixBackend,
    "stabilizer": StabilizerBackend,
    "ptm": PauliTransferMatrixBackend,
}

_CNOT_2Q = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=torch.cfloat,
)


def _make_backend(backend: str | QuantumBackend, n_qubits: int, **kwargs) -> QuantumBackend:
    if isinstance(backend, QuantumBackend):
        return backend
    return _BACKENDS[backend](n_qubits=n_qubits, **kwargs)


def _apply_backend_layer(
    layer: torch.nn.Module,
    backend: QuantumBackend,
    *,
    noise_model=None,
    noise_kwargs: dict | None = None,
    **kwargs,
) -> None:
    """Apply a built layer on a :class:`QuantumBackend`."""
    if isinstance(layer, BuiltNoiseChannel):
        opts = noise_kwargs or {}
        if hasattr(backend, "apply_noise_channel"):
            backend.apply_noise_channel(layer)
            return
        if isinstance(backend, MPSBackend):
            state = backend.state
            trajectory = opts.get("trajectory")
            if trajectory is None:
                trajectory = state.ndim <= 1
            backend.state = layer(state, trajectory=trajectory, rng=opts.get("rng"))
            return
        state = getattr(backend, "state", None)
        if state is not None:
            trajectory = opts.get("trajectory")
            if trajectory is None:
                trajectory = state.ndim <= 1
            backend.state = layer(state, trajectory=trajectory, rng=opts.get("rng"))
            return

        density = getattr(backend, "density", None)
        if density is not None:
            backend.density = layer(density)
            return

        rho = getattr(backend, "rho", None)
        if rho is not None:
            backend.rho = layer(rho)
            return

        raise NotImplementedError(
            "Noise channels require backend to expose a 'state', 'density', or 'rho' attribute for updates."
        )

    if isinstance(layer, operators.BuiltControlled):
        matrix, qubits = layer.backend_matrix_and_qubits(**kwargs)
        if len(qubits) == 2 and hasattr(backend, "apply_2q"):
            backend.apply_2q(matrix, qubits[0], qubits[1])
        else:
            if not hasattr(backend, "apply_dense"):
                raise NotImplementedError(
                    "Backend does not support applying multi-qubit dense gates."
                )
            backend.apply_dense(matrix, qubits)
    elif hasattr(layer, "qubit"):
        backend.apply_1q(layer.to_matrix(**kwargs), layer.qubit)
    elif isinstance(layer, operators.BuiltCNOT):
        if isinstance(backend, StateVectorBackend):
            backend.state = apply_CNOT(backend.state, layer.c, layer.t, backend.n_qubits)
        elif isinstance(backend, MPSBackend):
            gate = _CNOT_2Q.to(dtype=backend.dtype, device=backend.device)
            backend.apply_2q(gate, layer.c, layer.t)
        else:
            backend.apply_2q(layer.to_matrix(**kwargs), layer.c, layer.t)
    elif isinstance(layer, operators.BuiltCCNOT):
        if isinstance(backend, StateVectorBackend):
            backend.state = apply_CCNOT(
                backend.state, layer.c1, layer.c2, layer.t, backend.n_qubits
            )
        elif isinstance(backend, MPSBackend):
            raise NotImplementedError(
                "CCNOT is not supported by the MPS backend without dense control-gate matrices."
            )
        elif hasattr(backend, "apply_dense"):
            backend.apply_dense(layer.to_matrix(**kwargs), (layer.c1, layer.c2, layer.t))
        else:
            raise NotImplementedError("Backend does not provide a CCNOT implementation.")
    elif hasattr(layer, "c") and hasattr(layer, "t") and not hasattr(layer, "c2"):
        backend.apply_2q(layer.to_matrix(**kwargs), layer.c, layer.t)
    elif hasattr(layer, "a") and hasattr(layer, "b"):
        backend.apply_2q(layer.to_matrix(**kwargs), layer.a, layer.b)
    else:
        raise NotImplementedError("Only 1- and 2-qubit gates are supported.")


_ROTATION_TYPES = (operators.BuiltRX, operators.BuiltRY, operators.BuiltRZ)


def _is_self_inverse_matrix(matrix: torch.Tensor) -> bool:
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return torch.allclose(matrix @ matrix, eye, atol=1e-6, rtol=1e-6)


def _cancel_pair(prev: torch.nn.Module, curr: torch.nn.Module) -> bool:
    if type(prev) is not type(curr):
        return False
    if isinstance(prev, operators.BuiltCNOT):
        return prev.c == curr.c and prev.t == curr.t
    if isinstance(prev, operators.BuiltU):
        if getattr(prev, "qubit", None) != getattr(curr, "qubit", None):
            return False
        if not torch.allclose(prev.original_matrix, curr.original_matrix, atol=1e-6, rtol=1e-6):
            return False
        return _is_self_inverse_matrix(prev.original_matrix)
    return False


def _hydrate(special: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    matrix = torch.eye(1, dtype=special.dtype, device=special.device)
    eye = torch.eye(2, dtype=special.dtype, device=special.device)
    for i in range(num_qubits):
        matrix = torch.kron(matrix, special if i == qubit else eye)
    return matrix


def _combine_rotations(base: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    if base.dim() == 2 and new.dim() == 2:
        return new @ base
    if base.dim() == 2:
        base = base.unsqueeze(0).expand(new.shape[0], -1, -1)
    if new.dim() == 2:
        new = new.unsqueeze(0).expand(base.shape[0], -1, -1)
    if base.shape[0] == 1 and new.shape[0] > 1:
        base = base.expand_as(new)
    elif new.shape[0] == 1 and base.shape[0] > 1:
        new = new.expand_as(base)
    if base.shape[0] != new.shape[0]:
        raise ValueError("Mismatched batch sizes while combining rotations.")
    return torch.matmul(new, base)


class FusedSingleQubitRotation(torch.nn.Module):
    def __init__(
        self,
        axis_type: type[torch.nn.Module],
        layers: typing.List[torch.nn.Module],
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("FusedSingleQubitRotation requires at least one layer.")
        self.axis_type = axis_type
        self.layers = torch.nn.ModuleList(layers)
        first = layers[0]
        self.qubit = first.qubit
        self.num_qubits = first.num_qubits
        self.named = any(getattr(layer, "named", False) for layer in layers)
        if hasattr(first, "theta"):
            self._default_device = first.theta.device
        else:
            self._default_device = torch.device("cpu")

    def append(self, layer: torch.nn.Module) -> None:
        if not isinstance(layer, self.axis_type):
            raise TypeError("Cannot append layer of mismatched axis to fused rotation.")
        if getattr(layer, "qubit", None) != self.qubit:
            raise ValueError("Cannot append layer on different qubit to fused rotation.")
        self.layers.append(layer)
        self.named = self.named or getattr(layer, "named", False)

    def _reference_tensor(self, state: torch.Tensor | None) -> torch.Tensor:
        if state is not None:
            return state
        return torch.ones(1, dtype=torch.cfloat, device=self._default_device)

    def _rotation_from_layer(
        self,
        layer: torch.nn.Module,
        ref: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        a, b, a_op, b_op = layer.matrix_builder()
        if layer.named:
            if layer.name not in kwargs:
                raise KeyError(f"Missing value for named parameter '{layer.name}'.")
            t = kwargs[layer.name]
        else:
            t = layer.remapping(layer.theta)
        t = torch.as_tensor(t, device=ref.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        theta = (t / 2).view(-1, 1, 1)
        theta_t = torch.as_tensor(theta, device=ref.device, dtype=ref.dtype)
        a = a.to(device=ref.device, dtype=ref.dtype)
        b = b.to(device=ref.device, dtype=ref.dtype)
        rot = operators._to_like(a_op(theta_t), ref) * a + operators._to_like(b_op(theta_t), ref) * b
        if rot.dim() == 3 and rot.shape[0] == 1:
            rot = rot[0]
        return rot

    def _combined_rotation(self, state: torch.Tensor | None, kwargs: dict) -> torch.Tensor:
        ref = self._reference_tensor(state)
        combined: torch.Tensor | None = None
        for layer in self.layers:
            rot = self._rotation_from_layer(layer, ref, kwargs)
            combined = rot if combined is None else _combine_rotations(combined, rot)
        assert combined is not None
        return combined

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        rot = self._combined_rotation(state, kwargs)
        return operators.PairwiseRotation.apply(state, rot, self.qubit, self.num_qubits)

    def to_matrix(self, **kwargs) -> torch.Tensor:
        rot = self._combined_rotation(None, kwargs)
        if rot.dim() != 2:
            raise ValueError("to_matrix does not support batched named inputs for fused rotations.")
        return _hydrate(rot, self.qubit, self.num_qubits)

    def __str__(self) -> str:
        return f"Fused{self.axis_type.__name__}(q{self.qubit}, x{len(self.layers)})"


def _can_fuse(prev: torch.nn.Module, curr: torch.nn.Module) -> bool:
    if isinstance(prev, FusedSingleQubitRotation):
        return isinstance(curr, prev.axis_type) and getattr(curr, "qubit", None) == prev.qubit
    if isinstance(prev, _ROTATION_TYPES) and isinstance(curr, type(prev)):
        return prev.qubit == curr.qubit
    return False


def _fuse_layers(prev: torch.nn.Module, curr: torch.nn.Module) -> torch.nn.Module:
    if isinstance(prev, FusedSingleQubitRotation):
        prev.append(curr)
        return prev
    return FusedSingleQubitRotation(type(curr), [prev, curr])


def _optimize_layers(layers: typing.List[torch.nn.Module]) -> typing.List[torch.nn.Module]:
    optimized: list[torch.nn.Module] = []
    for layer in layers:
        if optimized and _cancel_pair(optimized[-1], layer):
            optimized.pop()
            continue
        if isinstance(layer, BuiltNoiseChannel):
            optimized.append(layer)
            continue
        if optimized and _can_fuse(optimized[-1], layer):
            optimized[-1] = _fuse_layers(optimized[-1], layer)
            continue
        optimized.append(layer)
    return optimized

__all__ = ["Circuit"]


class Circuit(torch.nn.Module):
    class DummyUnbuiltCircuit:
        """Helper class to decompose a circuit with subcircuits without building it."""

        def __init__(self, num_qubits: int, layers: typing.List[operators.Operator]):
            self.num_qubits = num_qubits
            self.layers = []
            for layer in layers:
                if hasattr(layer, "decompose"):
                    self.layers.extend(layer.decompose())  # type: ignore
                else:
                    self.layers.append(layer)

        def decompose(self):
            return self

    def __init__(
        self,
        layers: typing.List[operators.Operator],
        num_qubits=None,
        split_max_qubits=0,
        circuit=None,
    ):
        super().__init__()
        self.name = ""
        self.named = True
        if not hasattr(layers, "__iter__"):
            layers = [layers]
        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = len(self._used_qubits(layers))
        if circuit is not None:
            self.circuit = circuit
        else:
            if split_max_qubits > 0 and self.num_qubits > split_max_qubits:
                self.circuit = SplittedCircuit(
                    num_qubits=self.num_qubits,
                    subcircuitdict=splitter.split(
                        self.DummyUnbuiltCircuit(self.num_qubits, layers),
                        max_qubits=split_max_qubits,
                    ),
                )
            else:
                self.circuit = UnsplittedCircuit(self.num_qubits, layers)

    def forward(
        self,
        state=None,
        *,
        backend: str | QuantumBackend | None = None,
        backend_kwargs: dict | None = None,
        diff_method: str | None = None,
        noise_model=None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        noise_kwargs = kwargs.pop("noise_kwargs", None)
        circuit_to_run: "Circuit" = self if noise_model is None else noise_model.apply(self)

        if backend is not None:
            be = _make_backend(backend, self.num_qubits, **(backend_kwargs or {}))
            circ_impl = (
                circuit_to_run.circuit
                if isinstance(circuit_to_run.circuit, UnsplittedCircuit)
                else circuit_to_run.circuit.decompose()
            )
            be.allocate(self.num_qubits)
            if getattr(be, "requires_clifford", False):
                for layer in circ_impl.layers:
                    if isinstance(layer, BuiltNoiseChannel):
                        raise ValueError(
                            "Noise channels are not supported by the stabilizer backend."
                        )
                    if not hasattr(layer, "to_matrix"):
                        continue
                    try:
                        if hasattr(layer, "qubit"):
                            be.validate_1q_gate(layer.to_matrix(**kwargs), layer.qubit)
                        elif hasattr(layer, "c") and hasattr(layer, "t") and not hasattr(layer, "c2"):
                            be.validate_2q_gate(layer.to_matrix(**kwargs), layer.c, layer.t)
                        elif hasattr(layer, "a") and hasattr(layer, "b"):
                            be.validate_2q_gate(layer.to_matrix(**kwargs), layer.a, layer.b)
                    except ValueError as exc:
                        raise ValueError(
                            f"Layer {layer} cannot be simulated with the stabilizer backend because it is not Clifford."
                        ) from exc
            for layer in circ_impl.layers:
                _apply_backend_layer(
                    layer,
                    be,
                    noise_model=noise_model,
                    noise_kwargs=noise_kwargs,
                    **kwargs,
                )
            return be

        if diff_method == "adjoint":
            raise NotImplementedError(
                "diff_method='adjoint' is not supported yet; use 'parameter_shift' or autograd."
            )

        if diff_method == "parameter_shift":
            from .gradients import parameter_shift_forward

            return parameter_shift_forward(circuit_to_run.circuit, state, **kwargs)

        return circuit_to_run.circuit.forward(state, **kwargs)

    def to_matrix(self, **kwargs):
        return self.circuit.to_matrix(**kwargs)

    def __matmul__(self, x):
        """Allows the circuit to be multiplied with a state tensor, directly calling forward. Not compatible with named inputs."""
        return self.circuit.forward(x)

    @staticmethod
    def _used_qubits(layers: typing.List[operators.Operator]) -> typing.Set[int]:
        import qandle.measurements
        from qandle.noise.model import NoiseModel

        qubits = set()
        for layer in layers:
            if isinstance(layer, operators.CNOT):
                qubits.add(layer.c)
                qubits.add(layer.t)
            elif isinstance(layer, operators.SWAP):
                qubits.add(layer.a)
                qubits.add(layer.b)
            elif isinstance(layer, operators.Controlled):
                qubits.add(layer.c)
                target_qubits = tuple(NoiseModel._layer_qubits(layer.t))
                if not target_qubits and hasattr(layer.t, "num_qubits"):
                    target_qubits = tuple(range(layer.t.num_qubits))
                if not target_qubits:
                    raise ValueError(
                        "Unknown target for controlled layer, number of qubits could not be inferred."
                    )
                qubits.update(target_qubits)
            elif hasattr(layer, "qubit"):
                qubits.add(layer.qubit)  # type: ignore
            elif hasattr(layer, "num_qubits"):
                qubits.update(range(layer.num_qubits))  # type: ignore
            elif hasattr(layer, "qubits"):
                qubits.update(layer.qubits)
            elif isinstance(
                layer,
                (
                    qandle.measurements.UnbuiltMeasurement,
                    qandle.measurements.BuiltMeasurement,
                ),
            ):
                pass  # ignore measurements, as they act on all qubits anyway
            else:
                raise ValueError(
                    f"Unknown layer type {type(layer)}, number of qubits could not be inferred. Pass :code:`num_qubits` to the circuit."
                )
        if len(qubits) == 0:
            raise ValueError(
                "Number of qubits could not be inferred from layers. Please provide num_qubits to the circuit directly."
            )
        return qubits

    def draw(self):
        """Shorthand for drawer.draw(self.circuit)"""
        return drawer.draw(self.circuit)

    def split(self, max_qubits):
        if isinstance(self.circuit, UnsplittedCircuit):
            splitdict = splitter.split(self.circuit.decompose(), max_qubits=max_qubits)
            return Circuit(
                layers=[],
                num_qubits=self.num_qubits,
                circuit=SplittedCircuit(self.num_qubits, splitdict),
            )
        else:
            warnings.warn("Circuit already split, returning original circuit.")
            return self

    def decompose(self):
        return Circuit(layers=[], num_qubits=self.num_qubits, circuit=self.circuit.decompose())

    def to_qasm(self):
        return self.circuit.to_qasm()

    def to_openqasm2(self) -> str:
        return qasm.convert_to_qasm(self, qasm_version=2, include_header=True)

    def to_openqasm3(self) -> str:
        return qasm.convert_to_qasm(self, qasm_version=3, include_header=True)


class UnsplittedCircuit(torch.nn.Module):
    """
    Main class for quantum circuits. This class is a torch.nn.Module, so it can be used like any other PyTorch module.
    """

    def __init__(self, num_qubits: int, layers: list):
        """
        Create a new quantum circuit, building all layers.
        """
        super().__init__()
        self.num_qubits = num_qubits
        basis = F.one_hot(
            torch.tensor(0, dtype=torch.long), num_classes=2**num_qubits
        ).to(dtype=torch.complex64)
        self.state = basis
        layers_built = self._build_layers(layers, num_qubits)
        self.layers = torch.nn.ModuleList(layers_built)

    @staticmethod
    def _build_layers(layers: list, num_qubits: int) -> typing.List[torch.nn.Module]:
        layers_built = []
        for la in layers:
            if hasattr(la, "build"):
                layers_built.append(la.build(num_qubits=num_qubits))
            else:
                layers_built.append(la)
        return _optimize_layers(layers_built)

    def forward(self, state=None, **kwargs):
        """
        Run the circuit. If state is None, the initial state is used. Supply specific inputs to specific gates using the kwargs, of the form name:torch.Tensor
        """
        if state is None:
            state = self.state
        for mod in self.layers:
            if mod.named:
                state = mod(state, **kwargs)
            else:
                state = mod(state)
        return state

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = []
        for layer in self.layers:
            if hasattr(layer, "to_qasm"):
                reps.extend(layer.to_qasm())
        return reps  # type: ignore

    def decompose(self) -> "UnsplittedCircuit":
        """
        Decompose the circuit into a circuit without any subcircuits, returning a new, flat Circuit.
        """
        new_layers = []
        for layer in self.layers:
            if hasattr(layer, "decompose"):
                if isinstance(layer, (Circuit)):
                    new_layers.extend(layer.decompose().circuit.layers)
                else:
                    new_layers.extend(layer.decompose())
            else:
                new_layers.append(layer)
        return UnsplittedCircuit(self.num_qubits, new_layers)

    def split(self, **kwargs) -> typing.Union["SplittedCircuit", "UnsplittedCircuit"]:
        """
        Split the circuit into subcircuits, returning a CuttedCircuit if the circuit was split, or the original circuit if not.
        """
        if self.num_qubits == 1:
            warnings.warn("Can't split circuit with only 1 qubit")
            return self
        splitted = splitter.split(self, **kwargs)
        return SplittedCircuit(self.num_qubits, splitted)

    def to_matrix(self, **kwargs):
        """
        Get the matrix representation of the circuit.
        """
        return utils.reduce_dot(*[mod.to_matrix(**kwargs) for mod in self.layers])


class SplittedCircuit(UnsplittedCircuit):
    """
    Circuit with subcircuits, created by splitting a Circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        subcircuitdict: typing.Dict[int, splitter.SubcircuitContainer],
    ):
        torch.nn.Module.__init__(self)
        self.num_qubits = num_qubits
        basis = F.one_hot(
            torch.tensor(0, dtype=torch.long), num_classes=2**num_qubits
        ).to(dtype=torch.complex64)
        self.state = basis

        self._check_qubit_order(subcircuitdict)

        self.subcircuits = torch.nn.ModuleList(
            [
                Subcircuit(num_qubits, subcircuitdict[i].mapping, subcircuitdict[i].layers)
                for i in subcircuitdict
            ]
        )

    @staticmethod
    def _check_qubit_order(subcircuitdict):
        for sc in subcircuitdict.values():
            keys_sorted = sorted(sc.mapping.keys())
            values = [sc.mapping[k] for k in keys_sorted]
            assert values == sorted(values), f"Mapping {sc.mapping} is not ordered."

    def forward(self, inp=None, **kwargs):
        if inp is None:
            inp = self.state
        for sc in self.subcircuits:
            inp = sc(inp, **kwargs)
        return inp

    def to_qasm(self) -> qasm.QasmRepresentation:
        reps = [sc.to_qasm() for sc in self.subcircuits]
        return reps  # type: ignore

    def to_matrix(self, **kwargs):
        raise NotImplementedError("Can't get matrix representation of a splitted circuit.")


class Subcircuit(torch.nn.Module):
    """
    Submodule for CuttedCircuit, contains a single subcircuits
    """

    def __init__(self, num_qubits_parent: int, mapping: typing.Dict[int, int], layers: list):
        """
        num_qubits_parent: number of qubits in the parent circuit
        mapping: mapping of qubits from parent circuit to subcircuit
        layers: layers of the subcircuit
        """
        super().__init__()
        self.num_qubits_parent = num_qubits_parent
        self.mapping = mapping
        layers_built = UnsplittedCircuit._build_layers(layers, num_qubits=len(mapping))
        self.to_matrix, self.to_state = utils.get_matrix_transforms(
            num_qubits_parent, list(mapping.keys())
        )

        self.layers = torch.nn.ModuleList(layers_built)

    def forward(self, x, **kwargs):
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)
        x = self.to_matrix(x)
        for mod in self.layers:
            if mod.named:
                x = mod(x, **kwargs)
            else:
                x = mod(x)
        x = self.to_state(x)
        if not batched:
            x = x.squeeze(0)
        return x

    def to_qasm(self) -> qasm.QasmRepresentation:
        import re

        reps = [mod.to_qasm() for mod in self.layers if hasattr(mod, "to_qasm")]
        for r in reps:
            if r.qubit not in ("", None):
                r.qubit = [k for k, v in self.mapping.items() if v == r.qubit][0]
            else:
                rstr = r.gate_str

                def new_qubit(x):
                    old_q = int(x.group(1))
                    new_q = [k for k, v in self.mapping.items() if v == old_q][0]
                    return f"q[{new_q}]"

                rstr = re.sub(r"q\[(\d+)\]", new_qubit, rstr)
                r.gate_str = rstr
        return reps  # type: ignore
