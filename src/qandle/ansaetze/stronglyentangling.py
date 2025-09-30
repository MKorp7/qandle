import torch
import typing
import qw_map
import copy


import qandle.config as config
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm
from qandle.ansaetze.ansatz import UnbuiltAnsatz, BuiltAnsatz


__all__ = ["StronglyEntanglingLayer"]


def _ops_to_qasm(
    ops: typing.Iterable[op.Operator],
) -> typing.List[qasm.QasmRepresentation]:
    """Convert a sequence of operators to their QASM representations."""

    qasm_ops: typing.List[qasm.QasmRepresentation] = []
    for gate in ops:
        rep = gate.to_qasm()
        if isinstance(rep, qasm.QasmRepresentation):
            qasm_ops.append(rep)
        else:
            qasm_ops.extend(rep)
    return qasm_ops


class StronglyEntanglingLayer(UnbuiltAnsatz):
    """
    A strongly entangling layer, inspired by `this paper <https://arxiv.org/abs/1804.00633>`_.
    Consists of a series of single-qubit rotations on each qubit specified in the :code:`qubits` parameter, followed by a series of CNOT gates. A higher :code:`depth` will increase the expressivity of the circuit at the cost of more parameters and a linear increase in runtime.
    """

    def __init__(
        self,
        qubits: typing.List[int],
        num_qubits_total: typing.Union[int, None] = None,
        depth: int = 1,
        rotations=["rz", "ry", "rz"],
        q_params=None,
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
    ):
        super().__init__()
        self.depth = depth
        self.qubits = qubits
        self.num_qubits = num_qubits_total
        self.rots = [utils.parse_rot(r) for r in rotations]
        if q_params is None:
            q_params = torch.rand(depth, len(qubits), len(rotations))
        self.q_params = q_params
        if remapping is None:
            remapping = qw_map.identity
        self.remapping = remapping

    def build(self, *args, **kwargs) -> BuiltAnsatz:
        return StronglyEntanglingLayerBuilt(
            num_qubits=kwargs["num_qubits"],
            depth=self.depth,
            rotations=self.rots,
            q_params=self.q_params,
            remapping=self.remapping,
            qubits=self.qubits,
        )

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return _ops_to_qasm(self.decompose())

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers: typing.List[op.UnbuiltOperator] = []
        if not self.qubits:
            return layers

        num_qubits_total = self.num_qubits
        if num_qubits_total is None:
            try:
                num_qubits_total = max(self.qubits) + 1
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "StronglyEntanglingLayer.decompose requires at least one qubit to infer "
                    "the number of qubits."
                ) from exc

        rot_count = len(self.rots)
        for d in range(self.depth):
            for wi, w in enumerate(self.qubits):
                for r in range(rot_count):
                    layers.append(
                        self.rots[r](
                            qubit=w,
                            theta=self.q_params[d, wi, r],
                            remapping=self.remapping,
                        )
                    )
            if len(self.qubits) > 1:
                layers.extend(
                    StronglyEntanglingLayerBuilt._get_cnots(
                        self.qubits,
                        num_qubits_total,
                        d % (len(self.qubits) - 1),
                        build=False,
                    )
                )
        return layers


class StronglyEntanglingLayerBuilt(BuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        qubits: typing.List[int],
        depth: int,
        rotations: typing.List[op.UnbuiltParametrizedOperator],
        q_params: torch.Tensor,
        remapping: typing.Union[typing.Callable, None],
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.rots = rotations
        self.qubits = qubits
        layers = []
        for d in range(depth):
            for wi, w in enumerate(self.qubits):
                for r in range(len(self.rots)):
                    layers.append(
                        self.rots[r](
                            qubit=w,
                            theta=q_params[d, wi, r],
                            remapping=remapping,  # type: ignore
                        ).build(num_qubits)
                    )
            if len(self.qubits) > 1:
                layers.extend(
                    self._get_cnots(
                        self.qubits,
                        self.num_qubits,
                        d % (len(self.qubits) - 1),
                    )
                )
        self.mods = torch.nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.mods(state)

    @staticmethod
    def _get_cnots(
        qubits: typing.List[int],
        num_qubits_total: typing.Optional[int],
        iteration: int,
        *,
        build: bool = True,
    ) -> typing.List[typing.Union[op.UnbuiltOperator, op.BuiltOperator]]:
        if len(qubits) <= 1:
            return []

        if iteration < 0 or iteration >= len(qubits):
            raise ValueError("Iteration must be within the range of available qubits")

        cnots: typing.List[typing.Union[op.UnbuiltOperator, op.BuiltOperator]] = []
        for ci in range(len(qubits)):
            ti = (ci + iteration + 1) % len(qubits)
            cnot = op.CNOT(qubits[ci], qubits[ti])
            if build:
                if num_qubits_total is None:
                    raise ValueError("num_qubits_total must be provided to build CNOT gates")
                cnots.append(cnot.build(num_qubits_total))
            else:
                cnots.append(cnot)
        return cnots

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return _ops_to_qasm(self.decompose())

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers = []
        for mod in self.mods:
            layers.append(copy.deepcopy(mod))
        return layers