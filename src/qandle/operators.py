import torch
import abc
import typing
import inspect
import qw_map
import warnings

import qandle.config as config
import qandle.errors as errors
import qandle.qasm as qasm
import qandle.utils as utils
from qandle.kernels import apply_CCNOT, apply_CNOT

__all__ = [
    "Operator",
    "RX",
    "RY",
    "RZ",
    "CNOT",
    "CCNOT",
    "CZ",
    "Reset",
    "SWAP",
    "U",
    "CustomGate",
    "Controlled",
    "Invert",
    "BUILT_CLASS_RELATION",
]

matrixbuilder = typing.Tuple[torch.Tensor, torch.Tensor, typing.Callable, typing.Callable]


def _to_like(x: typing.Union[torch.Tensor, typing.Any], ref: torch.Tensor) -> torch.Tensor:
    """Return ``x`` as a tensor on the same device and with the same dtype as ``ref``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


class PairwiseRotation(torch.autograd.Function):
    """Apply a single-qubit rotation to all amplitude pairs of a state."""

    @staticmethod
    def forward(ctx, state: torch.Tensor, rot: torch.Tensor, qubit: int, num_qubits: int):
        # state: (..., 2**n)
        # rot: (2,2) or (batch,2,2)
        dim = state.shape[-1]
        state_flat = state.reshape(-1, dim)
        orig_state_batch = state_flat.shape[0]
        if rot.dim() == 2:
            rot_batch = rot.unsqueeze(0)
        else:
            rot_batch = rot
        orig_rot_batch = rot_batch.shape[0]
        batch = max(orig_state_batch, orig_rot_batch)
        if orig_state_batch != batch:
            state_flat = state_flat.expand(batch, -1)
        if orig_rot_batch != batch:
            rot_batch = rot_batch.expand(batch, -1, -1)
        out = state_flat.clone()
        step = 1 << (num_qubits - qubit - 1)
        for start in range(0, dim, 2 * step):
            s0 = slice(start, start + step)
            s1 = slice(start + step, start + 2 * step)
            in0 = state_flat[:, s0]
            in1 = state_flat[:, s1]
            r00 = rot_batch[:, 0, 0].unsqueeze(-1)
            r10 = rot_batch[:, 1, 0].unsqueeze(-1)
            r01 = rot_batch[:, 0, 1].unsqueeze(-1)
            r11 = rot_batch[:, 1, 1].unsqueeze(-1)
            out[:, s0] = in0 * r00 + in1 * r10
            out[:, s1] = in0 * r01 + in1 * r11
        ctx.save_for_backward(state_flat, rot_batch)
        ctx.qubit = qubit
        ctx.num_qubits = num_qubits
        ctx.state_shape = state.shape
        ctx.rot_shape = rot.shape
        ctx.orig_state_batch = orig_state_batch
        ctx.orig_rot_batch = orig_rot_batch
        return out.reshape_as(state).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state_flat, rot_batch = ctx.saved_tensors
        dim = state_flat.shape[-1]
        gflat = grad_output.reshape(-1, dim)
        step = 1 << (ctx.num_qubits - ctx.qubit - 1)
        grad_state = torch.zeros_like(gflat)
        grad_rot = torch.zeros_like(rot_batch)
        rot_adj = rot_batch.conj()
        for start in range(0, dim, 2 * step):
            s0 = slice(start, start + step)
            s1 = slice(start + step, start + 2 * step)
            go0 = gflat[:, s0]
            go1 = gflat[:, s1]
            in0 = state_flat[:, s0]
            in1 = state_flat[:, s1]
            r00 = rot_adj[:, 0, 0].unsqueeze(-1)
            r10 = rot_adj[:, 1, 0].unsqueeze(-1)
            r01 = rot_adj[:, 0, 1].unsqueeze(-1)
            r11 = rot_adj[:, 1, 1].unsqueeze(-1)
            grad_state[:, s0] = go0 * r00 + go1 * r01
            grad_state[:, s1] = go0 * r10 + go1 * r11
            grad_rot[:, 0, 0] += torch.sum(in0.conj() * go0, dim=-1)
            grad_rot[:, 1, 0] += torch.sum(in1.conj() * go0, dim=-1)
            grad_rot[:, 0, 1] += torch.sum(in0.conj() * go1, dim=-1)
            grad_rot[:, 1, 1] += torch.sum(in1.conj() * go1, dim=-1)
        # Reduce gradients if inputs were expanded
        if ctx.orig_rot_batch == 1 and grad_rot.shape[0] > 1:
            grad_rot = grad_rot.sum(dim=0, keepdim=True)
        elif ctx.rot_shape == torch.Size([2, 2]):
            grad_rot = grad_rot.sum(dim=0)
        if ctx.orig_state_batch == 1 and grad_state.shape[0] > 1:
            grad_state = grad_state.sum(dim=0, keepdim=True)
        grad_state = grad_state.reshape(ctx.state_shape)
        return grad_state, grad_rot, None, None


class AbstractNoForward(abc.ABCMeta, utils.do_not_implement("forward", "backward")):
    pass


class Operator(abc.ABC):
    """Everything that can be applied to a state."""

    named = False

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the operator."""

    def __repr__(self) -> str:
        return self.__str__()

    @abc.abstractmethod
    def to_qasm(self) -> qasm.QasmRepresentation:
        """Returns the OpenQASM2 representation of the operator."""


class UnbuiltOperator(Operator, abc.ABC):
    """Container class for operators that have not been built yet."""

    @abc.abstractmethod
    def build(self, num_qubits, **kwargs) -> "BuiltOperator":
        """Builds the operator, i.e. converts it to a torch.nn.Module."""


class BuiltOperator(Operator, torch.nn.Module, abc.ABC):
    """Container class for operators that have been built."""

    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Applies the operator to the state."""

    @abc.abstractmethod
    def to_matrix(self, **kwargs) -> torch.Tensor:
        """Returns the matrix representation of the operator, such that :code:`state @ matrix` is equivalent to :code:`forward(state)`. Might be significantly slower than forward."""


class U(UnbuiltOperator):
    def __init__(self, qubit: int, matrix: torch.Tensor):
        self.qubit = qubit
        self.matrix = matrix

    def __str__(self) -> str:
        return f"U_{self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str="U", qubit=self.qubit)

    def build(self, num_qubits, **kwargs) -> "BuiltU":
        return BuiltU(qubit=self.qubit, matrix=self.matrix, num_qubits=num_qubits)


class BuiltU(BuiltOperator):
    def __init__(
            self,
            qubit: int,
            matrix: torch.Tensor,
            num_qubits: int,
            self_description: str = "U",
    ):
        super().__init__()
        self.qubit = qubit
        self.num_qubits = num_qubits
        self.description = self_description

        self.original_matrix = matrix
        m = torch.eye(1)
        for i in range(self.num_qubits):
            m = torch.kron(m, matrix if i == self.qubit else torch.eye(2))
        self.matrix = m.to(torch.cfloat).contiguous()

    def __str__(self) -> str:
        return f"{self.description}_{self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        u = f"{self.original_matrix[0, 0]:.2f}, {self.original_matrix[0, 1]:.2f}, {self.original_matrix[1, 0]:.2f}, {self.original_matrix[1, 1]:.2f}"
        definition = (
                "gate "
                + self.description
                + "_"
                + str(self.qubit)
                + " q["
                + str(self.qubit)
                + "] {{ "
                + "U("
                + u
                + ") }}"
        )
        return qasm.QasmRepresentation(gate_str=definition, qubit=self.qubit)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.matrix

    def to_matrix(self, **kwargs) -> torch.Tensor:
        return self.matrix


CustomGate = BuiltU
"""
Define a custom single qubit gate. 
This module is always built, and therefore might have a big memory footprint. 

Attributes:
    qubit : int
        The index of the qubit this gate operates on.
    matrix : torch.Tensor
        The matrix representation of the gate. Must be a 2x2 unitary matrix.
    num_qubits : int
        The number of qubits in the circuit.
    self_description : str
        The description of the gate. Default is "U". This name will be used in the string representation of the gate and in the OpenQASM2 representation.

"""


class UnbuiltParametrizedOperator(UnbuiltOperator, metaclass=AbstractNoForward):
    """Container class for parametrized operators that have not been built yet."""

    def __init__(
            self,
            qubit: int,
            theta: typing.Union[float, torch.Tensor, None] = None,
            name: typing.Union[str, None] = None,
            **kwargs,
    ):
        """
        Creates a parametrized operator.

        """
        assert isinstance(qubit, int), "qubit must be an integer"
        assert qubit >= 0, "qubit must be >= 0"
        if isinstance(theta, (float, int)):
            theta = torch.tensor(theta, requires_grad=True, dtype=torch.float)
        remapping = kwargs.get("remapping", config.DEFAULT_MAPPING)
        if remapping is None:
            remapping = qw_map.none
        self.qubit = qubit
        self.name = name
        self.named = name is not None
        self.theta = theta
        self.remapping = remapping

    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.__class__.__name__}{self.qubit} ({self.name})"
        else:
            if self.theta is None:
                return f"{self.__class__.__name__}{self.qubit}"
            else:
                return f"{self.__class__.__name__}{self.qubit} ({self.remapping(self.theta).item():.2f})"

    def to_qasm(self) -> qasm.QasmRepresentation:
        if self.named:
            return qasm.QasmRepresentation(
                gate_str=self.__class__.__name__.lower(),
                qubit=self.qubit,
                qasm3_inputs=self.name,  # type: ignore # if name is None, named would be False
            )

        if self.theta is not None:
            return qasm.QasmRepresentation(
                gate_str=self.__class__.__name__.lower(),
                qubit=self.qubit,
                gate_value=self.remapping(self.theta).item(),
            )

        raise errors.UnbuiltGateError(
            "This gate has no parameter. Set parameter or build or set name before converting to OpenQASM2."
        )

    def build(self, num_qubits, **kwargs) -> "BuiltParametrizedOperator":
        return BUILT_CLASS_RELATION[self.__class__](
            qubit=self.qubit,
            initialtheta=self.theta,
            name=self.name,
            remapping=self.remapping,
            num_qubits=num_qubits,
        )


class BuiltParametrizedOperator(BuiltOperator, abc.ABC):
    def __init__(
            self,
            qubit: int,
            remapping: typing.Callable,
            num_qubits: int,
            initialtheta: typing.Union[torch.Tensor, None] = None,
            name: typing.Union[str, None] = None,
    ):
        super().__init__()
        self.qubit = qubit
        if initialtheta is None:
            initialtheta = torch.rand(1)
        self.name = name  # currently unused
        self.named = name is not None  # faster than checking if name is None every time
        self.theta = torch.nn.Parameter(initialtheta, requires_grad=True)
        self.remapping = remapping
        self.num_qubits = num_qubits
        self.unbuilt_class = BUILT_CLASS_RELATION.T[self.__class__]

        self.register_buffer("_i", torch.tensor(1j, dtype=torch.cfloat), persistent=False)

    def __str__(self) -> str:
        base = f"{self.unbuilt_class.__name__}_{self.qubit}"
        if self.named:
            return f"{base} ({self.name})"
        else:
            if self.theta is None:
                return f"{base}"
            else:
                return f"{base} ({self.remapping(self.theta).item():.2f})"

    def to_qasm(self) -> qasm.QasmRepresentation:
        if self.named:
            return qasm.QasmRepresentation(
                gate_str=self.unbuilt_class.__name__.lower(),
                qubit=self.qubit,
                qasm3_inputs=self.name,  # type: ignore # if name is None, named would be False
            )

        else:
            return qasm.QasmRepresentation(
                gate_str=self.unbuilt_class.__name__.lower(),
                qubit=self.qubit,
                gate_value=self.remapping(self.theta).item(),
            )

    def get_matrix(self, **kwargs) -> torch.Tensor:
        a, b, a_op, b_op = self.matrix_builder()
        if self.named:
            t = kwargs[self.name] / 2  # type: ignore # if name is None, named would be False
        else:
            t = self.remapping(self.theta) / 2
        t = torch.as_tensor(t, device=self.theta.device)  # ensure tensor & device
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        a = a.to(t.device)
        b = b.to(t.device)
        a_matrix = self.hydrated(a, device=t.device) * a_op(t)
        b_matrix = self.hydrated(b, device=t.device) * b_op(t)
        matrix = a_matrix + b_matrix
        return matrix

    def hydrated(
            self,
            special: typing.Union[torch.Tensor, None, typing.Tuple] = None,
            device: typing.Optional[torch.device] = None,
    ) -> torch.Tensor:
        if isinstance(special, torch.Tensor):
            device = special.device
            special = special.to(device=device, dtype=torch.cfloat)
        else:
            device = device or torch.device("cpu")
            if special is None:
                special = torch.eye(2, dtype=torch.cfloat, device=device)
            else:
                special = torch.tensor(special, dtype=torch.cfloat, device=device)
        matrix = torch.eye(1, dtype=torch.cfloat, device=device)
        for i in range(self.num_qubits):
            matrix = torch.kron(
                matrix,
                special if i == self.qubit else torch.eye(2, dtype=torch.cfloat, device=device),
            )
        return matrix

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        a, b, a_op, b_op = self.matrix_builder()
        if self.named:
            t = kwargs[self.name]
        else:
            t = self.remapping(self.theta)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        theta = (t / 2).view(-1, 1, 1)
        theta_t = torch.as_tensor(theta, device=state.device, dtype=state.dtype)
        a = a.to(device=state.device, dtype=state.dtype)
        b = b.to(device=state.device, dtype=state.dtype)
        rot = _to_like(a_op(theta_t), state) * a + _to_like(b_op(theta_t), state) * b
        if rot.shape[0] == 1:
            rot = rot[0]
        return PairwiseRotation.apply(state, rot, self.qubit, self.num_qubits)

    @abc.abstractmethod
    def matrix_builder(self) -> matrixbuilder:
        """Returns the matrix builder for the gate."""

    def to_matrix(self, **kwargs) -> torch.Tensor:
        return self.get_matrix(**kwargs)


class RX(UnbuiltParametrizedOperator):
    """
    Parametrized RX gate, i.e. a rotation around the x-axis of the Bloch sphere.

    This class represents a parametrized RX gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes:
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RX gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.identity` or :code:`lambda x: x`.
    """


class RY(UnbuiltParametrizedOperator):
    """
    Parametrized RY gate, i.e. a rotation around the y-axis of the Bloch sphere.

    This class represents a parametrized RY gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RY gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.identity` or :code:`lambda x: x`.
    """


class RZ(UnbuiltParametrizedOperator):
    """
    Parametrized RZ gate, i.e. a rotation around the z-axis of the Bloch sphere.

    This class represents a parametrized RZ gate in a quantum circuit.

    Used by :class:`qandle.qcircuit.QCircuit` to build the circuit.

    Attributes
        qubit : int
            The index of the qubit this gate operates on.
        theta : float, torch.Tensor, optional
            The parameter of the RZ gate, by default None. If None, a random parameter :math:`[0, 1]` is chosen.
        name : str, optional
            The name of the operator, by default None. If None, the operator is not named and does not accept named inputs.
        remapping : Callable, optional
            A function that remaps the parameter theta, by default config.DEFAULT_MAPPING. To disable remapping, pass :code:`qw_map.identity` or :code:`lambda x: x`.
    """


class BuiltRX(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            torch.tensor([[0, -1j], [-1j, 0]], dtype=torch.cfloat),
            torch.eye(2, dtype=torch.cfloat),
            torch.sin,
            torch.cos,
        )


class BuiltRY(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            torch.tensor([[0, -1], [1, 0]], dtype=torch.cfloat),
            torch.eye(2, dtype=torch.cfloat),
            torch.sin,
            torch.cos,
        )


class BuiltRZ(BuiltParametrizedOperator):
    def matrix_builder(self) -> matrixbuilder:
        return (
            torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat),
            torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat),
            lambda theta: torch.exp(-self._i * theta),
            lambda theta: torch.exp(self._i * theta),
        )


class CNOT(UnbuiltOperator):
    """CNOT gate."""

    def __init__(self, control: int, target: int):
        assert control != target, "Control and target must be different"
        self.c = control
        self.t = target

    def __str__(self) -> str:
        return f"CNOT {self.c}|{self.t}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"cx q[{self.c}], q[{self.t}]")

    def build(self, num_qubits, **kwargs) -> "BuiltCNOT":
        return BuiltCNOT(control=self.c, target=self.t, num_qubits=num_qubits)


class CCNOT(UnbuiltOperator):
    """CCNOT gate."""

    def __init__(self, control1: int, control2: int, target: int):
        assert control1 != target, "Control and target must be different"
        assert control2 != target, "Control and target must be different"
        assert control1 != control2, "Controls must be different"
        self.c1 = control1
        self.c2 = control2
        self.t = target

    def __str__(self) -> str:
        return f"CCNOT {self.c1}|{self.c2}|{self.t}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"ccx q[{self.c1}], q[{self.c2}], q[{self.t}]")

    def build(self, num_qubits, **kwargs) -> "BuiltCCNOT":
        return BuiltCCNOT(control1=self.c1, control2=self.c2, target=self.t, num_qubits=num_qubits)


class Invert(UnbuiltOperator):
    """
    Special class for inverted gates. Apply the inverse of the target operator.
    """

    def __init__(self, target: Operator):
        self.t = target

    def __str__(self) -> str:
        return f"{self.t}^-1"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"{self.t}^-1")

    def build(self, num_qubits, **kwargs) -> "BuiltInvert":
        return BuiltInvert(target=self.t, num_qubits=num_qubits)


class BuiltInvert(BuiltOperator):
    def __init__(self, target: Operator, num_qubits: int):
        super().__init__()
        if hasattr(target, "build"):
            warnings.warn(
                "Building target operator in Invert. If the target operator is parameterized, this might lead to different parameter initialization. Please build the target operator before inverting it."
            )
            target = target.build(num_qubits)
        self.target = target
        self.num_qubits = num_qubits

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        target_matrix = self.target.to_matrix()
        return state @ torch.linalg.inv(target_matrix)

    def __str__(self) -> str:
        return Invert(self.target).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return Invert(self.target).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:
        return torch.linalg.inv(self.target.to_matrix())


class Controlled(UnbuiltOperator):
    """
    Special class for controlled gates. Use a control qubit, and apply the target operator if the control qubit is 1.
    """

    def __init__(self, control: int, target: Operator):
        self.c = control
        self.t = target

    def __str__(self) -> str:
        return f"Controlled {self.c}|{self.t}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"controlled q[{self.c}], {self.t}")

    def build(self, num_qubits, **kwargs) -> "BuiltControlled":
        return BuiltControlled(control=self.c, target=self.t, num_qubits=num_qubits)


class BuiltControlled(BuiltOperator):
    def __init__(self, control: int, target: Operator, num_qubits: int):
        super().__init__()
        self.c = control
        if hasattr(target, "build"):
            target = target.build(num_qubits)
        self.t = target
        self.named = target.named
        self.num_qubits = num_qubits
        self._target_qubits = self._extract_gate_qubits(self.t)
        if not self._target_qubits:
            raise ValueError("Controlled target does not expose any qubits.")
        if self.c in self._target_qubits:
            raise ValueError("Control qubit overlaps with target operator qubits.")
        self.controlled_qubits = [self.c, *self._target_qubits]
        self._target_basis_indices = self._basis_indices(self._target_qubits)
        self._target_accepts_kwargs = self._accepts_kwargs(self.t)
        self._backend_perm_inv = self._compute_backend_permutation(len(self.controlled_qubits))

    @staticmethod
    def _accepts_kwargs(module: torch.nn.Module) -> bool:
        try:
            signature = inspect.signature(module.forward)
        except (ValueError, AttributeError):
            return False
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _extract_gate_qubits(gate, _visited=None) -> list[int]:
        if _visited is None:
            _visited = set()
        obj_id = id(gate)
        if obj_id in _visited:
            return []
        _visited.add(obj_id)

        qubits: list[int] = []

        if hasattr(gate, "qubits"):
            qubits.extend(int(q) for q in gate.qubits)

        potential_attrs = (
            "qubit",
            "c",
            "t",
            "a",
            "b",
            "c1",
            "c2",
            "control",
            "target",
            "control1",
            "control2",
        )
        for attr in potential_attrs:
            if hasattr(gate, attr):
                value = getattr(gate, attr)
                qubits.extend(
                    BuiltControlled._collect_qubits_from_value(value, _visited)
                )

        seen: set[int] = set()
        ordered: list[int] = []
        for q in qubits:
            if q not in seen:
                seen.add(q)
                ordered.append(q)
        return ordered

    @staticmethod
    def _collect_qubits_from_value(value, visited) -> list[int]:
        collected: list[int] = []
        if isinstance(value, int):
            collected.append(value)
        elif isinstance(value, (list, tuple, set)):
            for elem in value:
                collected.extend(BuiltControlled._collect_qubits_from_value(elem, visited))
        elif isinstance(value, dict):
            for elem in value.values():
                collected.extend(BuiltControlled._collect_qubits_from_value(elem, visited))
        elif hasattr(value, "qubit") or hasattr(value, "qubits"):
            collected.extend(BuiltControlled._extract_gate_qubits(value, visited))
        return collected

    def _basis_indices(self, qubits: list[int]) -> list[int]:
        indices: list[int] = []
        num_bits = len(qubits)
        for local_index in range(1 << num_bits):
            global_index = 0
            for pos, qubit in enumerate(qubits):
                bit = (local_index >> (num_bits - pos - 1)) & 1
                if bit:
                    global_index |= 1 << (self.num_qubits - qubit - 1)
            indices.append(global_index)
        return indices

    @staticmethod
    def _reverse_bits(index: int, width: int) -> int:
        result = 0
        for _ in range(width):
            result = (result << 1) | (index & 1)
            index >>= 1
        return result

    @classmethod
    def _compute_backend_permutation(cls, width: int) -> list[int]:
        if width == 0:
            return [0]
        perm = [cls._reverse_bits(i, width) for i in range(1 << width)]
        perm_inv = [0] * (1 << width)
        for big, little in enumerate(perm):
            perm_inv[little] = big
        return perm_inv

    def _infer_state_device(self) -> torch.device:
        for tensor in self.t.parameters(recurse=True):
            return tensor.device
        for tensor in self.t.buffers(recurse=True):
            return tensor.device
        return torch.device("cpu")

    def _target_matrix(self, **kwargs) -> torch.Tensor:
        device = self._infer_state_device()
        dtype = torch.complex64
        dim = len(self._target_basis_indices)
        matrix = torch.zeros((dim, dim), dtype=dtype, device=device)
        selector = torch.tensor(self._target_basis_indices, dtype=torch.long, device=device)
        for col, global_index in enumerate(self._target_basis_indices):
            state = torch.zeros(2 ** self.num_qubits, dtype=dtype, device=device)
            state[global_index] = 1
            if self._target_accepts_kwargs:
                out = self.t(state, **kwargs)
            else:
                out = self.t(state)
            matrix[:, col] = out.index_select(0, selector)
        return matrix

    def controlled_matrix(self, **kwargs) -> torch.Tensor:
        target_matrix = self._target_matrix(**kwargs)
        dim = target_matrix.shape[0]
        matrix = torch.eye(2 * dim, dtype=target_matrix.dtype, device=target_matrix.device)
        matrix[dim:, dim:] = target_matrix
        return matrix

    def _embed_matrix(self, local_matrix: torch.Tensor, qubits: list[int]) -> torch.Tensor:
        rest_qubits = [q for q in range(self.num_qubits) if q not in qubits]
        rest_dim = 2 ** len(rest_qubits)
        embedded = torch.kron(
            torch.eye(rest_dim, dtype=local_matrix.dtype, device=local_matrix.device),
            local_matrix,
        )
        total = self.num_qubits
        tensor = embedded.reshape([2] * (2 * total))
        order = rest_qubits + qubits
        perm = [order.index(q) for q in range(total)]
        tensor = tensor.permute(perm + [total + p for p in perm])
        return tensor.reshape(2 ** total, 2 ** total)

    def backend_matrix_and_qubits(self, **kwargs) -> tuple[torch.Tensor, list[int]]:
        local = self.controlled_matrix(**kwargs)
        indices = torch.tensor(
            self._backend_perm_inv,
            dtype=torch.long,
            device=local.device,
        )
        backend_matrix = local.index_select(0, indices).index_select(1, indices)
        backend_qubits = list(reversed(self.controlled_qubits))
        return backend_matrix, backend_qubits

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        target_matrix = self.t.to_matrix(**kwargs)
        c2 = self.num_qubits - self.c - 1
        mask = 1 << c2
        dim = 2 ** self.num_qubits
        device = state.device
        indices = torch.arange(dim, device=device)
        c1_mask = (indices & mask) != 0
        batched = state.dim() == 1
        if batched:
            state = state.unsqueeze(0)
        state_c1 = state.clone()
        state_c1[:, ~c1_mask] = 0
        transformed_c1 = torch.squeeze(state_c1.unsqueeze(1) @ target_matrix, 1)
        out = torch.where(c1_mask, transformed_c1, state)
        if batched:
            out = out.squeeze(0)
        return out

    def __str__(self) -> str:
        return Controlled(self.c, self.t).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return Controlled(self.c, self.t).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:

        target_matrix = self.t.to_matrix(**kwargs)
        dim = target_matrix.shape[-1]
        device = target_matrix.device
        dtype = target_matrix.dtype
        control_be = self.num_qubits - self.c - 1
        mask = 1 << control_be
        indices = torch.arange(dim, device=device)
        control_on = (indices & mask) != 0
        result = torch.eye(dim, dtype=dtype, device=device)
        control_idx = torch.nonzero(control_on, as_tuple=False).squeeze(-1)
        if control_idx.numel() > 0:
            result[control_idx.unsqueeze(1), control_idx.unsqueeze(0)] = target_matrix[
                control_idx.unsqueeze(1), control_idx.unsqueeze(0)
            ]
        return result


class CZ(UnbuiltOperator):
    """CZ gate."""

    def __init__(self, control: int, target: int):
        assert control != target, "Control and target must be different"
        self.c = control
        self.t = target

    def __str__(self) -> str:
        return f"CZ {self.c}|{self.t}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"cz q[{self.c}], q[{self.t}]")

    def build(self, num_qubits, **kwargs) -> "BuiltCZ":
        return BuiltCZ(control=self.c, target=self.t, num_qubits=num_qubits)


class BuiltCNOT(BuiltOperator):
    def __init__(self, control: int, target: int, num_qubits: int):
        super().__init__()
        self.c = control
        self.t = target
        self.num_qubits = num_qubits
        self._use_dense = config.USE_DENSE_CONTROL_GATES
        dense = (
            self._calculate_matrix(control, target, num_qubits)
            if self._use_dense
            else torch.empty(0, dtype=torch.cfloat)
        )
        self.register_buffer("_dense_matrix", dense, persistent=False)

    @staticmethod
    def _calculate_matrix(c: int, t: int, num_qubits: int):
        dim = 2 ** num_qubits
        M = torch.zeros((dim, dim), dtype=torch.cfloat)
        c2, t2 = num_qubits - c - 1, num_qubits - t - 1
        for i in range(2 ** num_qubits):
            if i & (1 << c2):
                M[i, i ^ (1 << t2)] = 1
            else:
                M[i, i] = 1
        return M

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self._use_dense:
            return state @ self._dense_matrix
        return apply_CNOT(state, self.c, self.t, self.num_qubits)

    def __str__(self) -> str:
        return CNOT(self.c, self.t).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return CNOT(self.c, self.t).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:
        if self._use_dense:
            return self._dense_matrix
        mat = self._calculate_matrix(self.c, self.t, self.num_qubits)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if device is not None or dtype is not None:
            mat = mat.to(device=device or mat.device, dtype=dtype or mat.dtype)
        return mat


class BuiltCCNOT(BuiltOperator):
    def __init__(self, control1: int, control2: int, target: int, num_qubits: int):
        super().__init__()
        self.c1 = control1
        self.c2 = control2
        self.t = target
        self.num_qubits = num_qubits
        self._use_dense = config.USE_DENSE_CONTROL_GATES
        dense = (
            self._calculate_matrix(control1, control2, target, num_qubits)
            if self._use_dense
            else torch.empty(0, dtype=torch.cfloat)
        )
        self.register_buffer("_dense_matrix", dense, persistent=False)

    @staticmethod
    def _calculate_matrix(c1: int, c2: int, t: int, num_qubits: int):
        dim = 2 ** num_qubits
        indices = torch.arange(dim)
        M = torch.zeros((dim, dim), dtype=torch.cfloat)
        # convention to big-endian
        c1 = num_qubits - c1 - 1
        c2 = num_qubits - c2 - 1
        t = num_qubits - t - 1

        control1_mask = 1 << c1
        control2_mask = 1 << c2
        target_mask = 1 << t

        are_controls_on = ((indices & control1_mask) != 0) & ((indices & control2_mask) != 0)
        source_indices = indices
        target_indices = torch.where(are_controls_on, indices ^ target_mask, indices)

        M[source_indices, target_indices] = 1
        return M

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self._use_dense:
            return state @ self._dense_matrix
        return apply_CCNOT(state, self.c1, self.c2, self.t, self.num_qubits)

    def __str__(self) -> str:
        return CCNOT(self.c1, self.c2, self.t).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return CCNOT(self.c1, self.c2, self.t).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:
        if self._use_dense:
            return self._dense_matrix
        mat = self._calculate_matrix(self.c1, self.c2, self.t, self.num_qubits)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if device is not None or dtype is not None:
            mat = mat.to(device=device or mat.device, dtype=dtype or mat.dtype)
        return mat


class BuiltCZ(BuiltOperator):
    def __init__(self, control: int, target: int, num_qubits: int):
        super().__init__()
        self.c = control
        self.t = target
        self.num_qubits = num_qubits
        self.register_buffer(
            "_M", self._calculate_matrix(control, target, num_qubits), persistent=False
        )

    @staticmethod
    def _calculate_matrix(c: int, t: int, num_qubits: int):
        c2, t2 = num_qubits - c - 1, num_qubits - t - 1
        indices = torch.arange(2 ** num_qubits)
        diag = torch.ones(2 ** num_qubits, dtype=torch.cfloat)
        mask = ((indices & (1 << c2)) != 0) & ((indices & (1 << t2)) != 0)
        diag[mask] = -1
        M = torch.diag(diag)
        return M

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self._M

    def __str__(self) -> str:
        return CZ(self.c, self.t).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return CZ(self.c, self.t).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:
        return self._M


class BuiltReset(BuiltOperator):
    def __init__(self, qubit: int, num_qubits: int):
        super().__init__()
        self.qubit = qubit
        self.num_qubits = num_qubits
        self.to_matrix_transform, self.to_state_transform = utils.get_matrix_transforms(
            num_qubits, [qubit]
        )

    def forward(self, state: torch.Tensor):
        unbatched = state.dim() == 1
        if unbatched:
            state = state.unsqueeze(0)
        state = self.to_matrix_transform(state)
        old_norm = torch.linalg.norm(state, dim=-1)
        old_norm_0 = (state[:, 0].abs()) + 1e-5
        scale = old_norm / old_norm_0
        new_state = torch.zeros_like(state, dtype=torch.cfloat)
        new_state[:, 0] = state[:, 0] * scale
        state = self.to_state_transform(new_state)
        state = state / torch.linalg.norm(state, dim=-1, keepdim=True)
        if unbatched:
            state = state.squeeze(0)
        return state

    def __str__(self) -> str:
        return Reset(self.qubit).__str__()

    def to_qasm(self) -> qasm.QasmRepresentation:
        return Reset(self.qubit).to_qasm()

    def to_matrix(self, **kwargs) -> torch.Tensor:
        raise ValueError("Reset gate does not have a matrix representation.")


class Reset(UnbuiltOperator):
    """
    Reset gate. Resets the selected qubit to the |0> state, while preserving the norm of the state.
    """

    def __init__(self, qubit: int):
        self.qubit = qubit

    def __str__(self) -> str:
        return f"Reset {self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str="reset", qubit=self.qubit)

    def build(self, num_qubits, **kwargs) -> "BuiltReset":
        return BUILT_CLASS_RELATION[self.__class__](qubit=self.qubit, num_qubits=num_qubits)


class BuiltSWAP(BuiltOperator):
    def __init__(self, a: int, b: int, num_qubits: int):
        super().__init__()
        self.a = a
        self.b = b
        self.num_qubits = num_qubits
        self.register_buffer("_M", self._calculate_matrix(a, b, num_qubits), persistent=False)

    @staticmethod
    def _calculate_matrix(a: int, b: int, num_qubits: int):
        swap_matrix = torch.eye(2 ** num_qubits)
        a, b = num_qubits - a - 1, num_qubits - b - 1
        for i in range(2 ** num_qubits):
            swapped_i = i
            if ((i >> a) & 1) != ((i >> b) & 1):
                swapped_i = i ^ ((1 << a) | (1 << b))
            swap_matrix[i, i] = 0
            swap_matrix[i, swapped_i] = 1
        return swap_matrix.to(torch.cfloat)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self._M

    def __str__(self) -> str:
        return f"SWAP {self.a}|{self.b}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"swap q[{self.a}], q[{self.b}]")

    def to_matrix(self, **kwargs) -> torch.Tensor:
        return self._M


class SWAP(UnbuiltOperator):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"Swap {self.a}|{self.b}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"swap q[{self.a}], q[{self.b}]")

    def build(self, num_qubits, **kwargs) -> "BuiltSWAP":
        return BuiltSWAP(a=self.a, b=self.b, num_qubits=num_qubits)


class rdict(dict):
    """reversible dict"""

    @property
    def T(self):
        return {v: k for k, v in self.items()}


BUILT_CLASS_RELATION = rdict(
    {
        UnbuiltOperator: BuiltOperator,
        UnbuiltParametrizedOperator: BuiltParametrizedOperator,
        RX: BuiltRX,
        RY: BuiltRY,
        RZ: BuiltRZ,
        CNOT: BuiltCNOT,
        CCNOT: BuiltCCNOT,
        Reset: BuiltReset,
        U: BuiltU,
        SWAP: BuiltSWAP,
        CZ: BuiltCZ,
        Invert: BuiltInvert,
        Controlled: BuiltControlled,
    }
)