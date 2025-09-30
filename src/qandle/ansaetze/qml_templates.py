"""Parameterized quantum circuit templates for quantum machine learning."""

from __future__ import annotations

import typing

import torch

from qandle.qcircuit import Circuit

import qandle.operators as op

__all__ = [
    "HardwareEfficientAnsatz",
    "StronglyEntanglingLayers",
    "QAOALayer",
]


LayerInputs = typing.Dict[str, torch.Tensor]
LayerSeq = typing.Sequence[typing.Union[op.Operator, torch.nn.Module]]


def _append_layers(
    circuit: typing.Optional[Circuit],
    layers: LayerSeq,
    *,
    num_qubits: int,
) -> Circuit:
    if circuit is None:
        return Circuit(layers=list(layers), num_qubits=num_qubits)

    if circuit.num_qubits != num_qubits:
        raise ValueError(
            f"Circuit acts on {circuit.num_qubits} qubits, but {num_qubits} were requested."
        )

    existing_layers = list(circuit.circuit.layers)
    existing_layers.extend(layers)
    return Circuit(layers=existing_layers, num_qubits=num_qubits)


def _generate_edges(
    n_qubits: int,
    *,
    entangler: str,
    edges: typing.Optional[typing.Sequence[typing.Tuple[int, int]]],
) -> typing.List[typing.Tuple[int, int]]:
    if edges is not None:
        return list(edges)

    if entangler not in {"line", "ring"}:
        raise ValueError("entangler must be 'line' or 'ring'.")

    if entangler == "line":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]


class HardwareEfficientAnsatz(torch.nn.Module):
    r"""Hardware-efficient ansatz constructed from local Euler rotations and entanglers.

    The unitary implemented by :math:`L` layers is

    .. math::

        U_{\text{HEA}}(\Theta) = \prod_{\ell=1}^{L} U_{\text{ent}}^{(\ell)}
        \prod_{i=0}^{n-1} R_Z^{(i)}(\gamma_{\ell, i}) R_Y^{(i)}(\beta_{\ell, i})
        R_Z^{(i)}(\alpha_{\ell, i}).

    The ansatz exposes its parameters via the dictionary returned from
    :meth:`forward`, which maps rotation names to the corresponding learnable
    angles.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        *,
        entangler: str = "ring",
        edges: typing.Optional[typing.Sequence[typing.Tuple[int, int]]] = None,
        entangler_gate: str = "cz",
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be a positive integer.")
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")

        entangler_gate = entangler_gate.lower()
        if entangler_gate not in {"cz", "cnot"}:
            raise ValueError("entangler_gate must be 'cz' or 'cnot'.")

        self.n_qubits = n_qubits
        self.layers = layers
        self.entangler = entangler
        self.entangler_gate = entangler_gate
        self.edges = _generate_edges(n_qubits, entangler=entangler, edges=edges)
        self.theta = torch.nn.Parameter(
            torch.randn(layers, n_qubits, 3) * 0.1, requires_grad=True
        )

    @property
    def _entangler_gate(self) -> typing.Callable[[int, int], op.UnbuiltOperator]:
        return op.CZ if self.entangler_gate == "cz" else op.CNOT

    def forward(
        self, circuit: typing.Optional[Circuit] = None
    ) -> typing.Tuple[Circuit, LayerInputs]:
        layers: typing.List[op.UnbuiltOperator] = []
        inputs: LayerInputs = {}
        gate_cls = self._entangler_gate

        for layer_idx in range(self.layers):
            for qubit in range(self.n_qubits):
                names = (
                    f"hea_l{layer_idx}_q{qubit}_rz0",
                    f"hea_l{layer_idx}_q{qubit}_ry",
                    f"hea_l{layer_idx}_q{qubit}_rz1",
                )
                rotations = (
                    op.RZ(qubit=qubit, name=names[0]),
                    op.RY(qubit=qubit, name=names[1]),
                    op.RZ(qubit=qubit, name=names[2]),
                )
                for idx, rotation in enumerate(rotations):
                    layers.append(rotation)
                    inputs[names[idx]] = self.theta[layer_idx, qubit, idx]

            for control, target in self.edges:
                layers.append(gate_cls(control, target))

        circuit_out = _append_layers(circuit, layers, num_qubits=self.n_qubits)
        return circuit_out, inputs


class StronglyEntanglingLayers(torch.nn.Module):
    r"""Strongly entangling layers with alternating :math:`ZZ` couplings.

    Each layer performs local :math:`R_Z`, :math:`R_Y`, :math:`R_Z` rotations on
    all qubits followed by two rounds of :math:`ZZ` interactions. The first round
    couples even-labelled qubits and the second round couples odd-labelled
    qubits, yielding the pattern introduced in
    :cite:`sim2019expressibility`.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        *,
        ring: bool = True,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be a positive integer.")
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")

        self.n_qubits = n_qubits
        self.layers = layers
        self.ring = ring
        self.theta = torch.nn.Parameter(
            torch.randn(layers, n_qubits, 3) * 0.1, requires_grad=True
        )
        self.lam = torch.nn.Parameter(torch.randn(layers, n_qubits) * 0.1, requires_grad=True)

    def _neighbor(self, qubit: int) -> typing.Optional[int]:
        nxt = (qubit + 1) % self.n_qubits if self.ring else qubit + 1
        if not self.ring and nxt >= self.n_qubits:
            return None
        return nxt

    def forward(
        self, circuit: typing.Optional[Circuit] = None
    ) -> typing.Tuple[Circuit, LayerInputs]:
        layers: typing.List[op.UnbuiltOperator] = []
        inputs: LayerInputs = {}

        for layer_idx in range(self.layers):
            for qubit in range(self.n_qubits):
                names = (
                    f"sel_l{layer_idx}_q{qubit}_rz0",
                    f"sel_l{layer_idx}_q{qubit}_ry",
                    f"sel_l{layer_idx}_q{qubit}_rz1",
                )
                rotations = (
                    op.RZ(qubit=qubit, name=names[0]),
                    op.RY(qubit=qubit, name=names[1]),
                    op.RZ(qubit=qubit, name=names[2]),
                )
                for idx, rotation in enumerate(rotations):
                    layers.append(rotation)
                    inputs[names[idx]] = self.theta[layer_idx, qubit, idx]

            for parity in (0, 1):
                for qubit in range(parity, self.n_qubits, 2):
                    partner = self._neighbor(qubit)
                    if partner is None:
                        continue
                    name = f"sel_l{layer_idx}_p{parity}_q{qubit}_zz"
                    layers.append(op.CNOT(qubit, partner))
                    layers.append(op.RZ(qubit=partner, name=name))
                    inputs[name] = self.lam[layer_idx, qubit]
                    layers.append(op.CNOT(qubit, partner))

        circuit_out = _append_layers(circuit, layers, num_qubits=self.n_qubits)
        return circuit_out, inputs


class QAOALayer(torch.nn.Module):
    r"""Single layer of the Quantum Approximate Optimisation Algorithm (QAOA).

    The cost Hamiltonian is specified by Ising :math:`ZZ` couplings with weights
    ``w`` and local fields ``h``:

    .. math::

        H_C = \sum_{(i, j) \in E} w_{ij} Z_i Z_j + \sum_i h_i Z_i.

    The mixing Hamiltonian is :math:`H_M = \sum_i X_i`. Each layer applies
    :math:`U_M(\beta_k) U_C(\gamma_k)` with

    .. math::

        U_C(\gamma_k) = \prod_{(i, j)} ZZ(2 \gamma_k w_{ij})
        \prod_i R_Z(2 \gamma_k h_i), \qquad
        U_M(\beta_k) = \prod_i R_X(2 \beta_k).
    """

    def __init__(
        self,
        n_qubits: int,
        p: int,
        edges: typing.Sequence[typing.Tuple[int, int]],
        w: torch.Tensor,
        *,
        h: typing.Optional[torch.Tensor] = None,
        learn_params: bool = True,
    ) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("p must be positive.")
        if len(edges) != len(w):
            raise ValueError("Edge weights must match the number of edges.")

        self.n_qubits = n_qubits
        self.p = p
        self.edges = list(edges)
        self.learn_params = learn_params
        self.register_buffer("w", w.to(dtype=torch.float))
        if h is not None:
            if len(h) != n_qubits:
                raise ValueError("Local field vector must have length n_qubits.")
            self.register_buffer("h", h.to(dtype=torch.float))
        else:
            self.register_buffer("h", torch.zeros(n_qubits, dtype=torch.float))

        if learn_params:
            self.gamma = torch.nn.Parameter(torch.zeros(p, dtype=torch.float))
            self.beta = torch.nn.Parameter(torch.zeros(p, dtype=torch.float))
        else:
            self.gamma = None
            self.beta = None

    def _get_parameters(
        self,
        gamma: typing.Optional[torch.Tensor],
        beta: typing.Optional[torch.Tensor],
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.learn_params:
            assert self.gamma is not None and self.beta is not None
            return self.gamma, self.beta

        if gamma is None or beta is None:
            raise ValueError("gamma and beta must be provided when learn_params=False.")
        gamma_t = torch.as_tensor(gamma, dtype=torch.float)
        beta_t = torch.as_tensor(beta, dtype=torch.float)
        if gamma_t.shape != (self.p,) or beta_t.shape != (self.p,):
            raise ValueError("gamma and beta must have shape (p,).")
        return gamma_t, beta_t

    def forward(
        self,
        circuit: typing.Optional[Circuit] = None,
        *,
        gamma: typing.Optional[torch.Tensor] = None,
        beta: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[Circuit, LayerInputs]:
        gamma_vec, beta_vec = self._get_parameters(gamma, beta)

        layers: typing.List[op.UnbuiltOperator] = []
        inputs: LayerInputs = {}

        w = self.w.to(device=gamma_vec.device, dtype=gamma_vec.dtype)
        h = self.h.to(device=gamma_vec.device, dtype=gamma_vec.dtype)

        for layer_idx in range(self.p):
            gamma_val = gamma_vec[layer_idx]
            beta_val = beta_vec[layer_idx]

            for edge_idx, (control, target) in enumerate(self.edges):
                name = f"qaoa_l{layer_idx}_edge{edge_idx}_zz"
                layers.append(op.CNOT(control, target))
                layers.append(op.RZ(qubit=target, name=name))
                inputs[name] = 2.0 * gamma_val * w[edge_idx]
                layers.append(op.CNOT(control, target))

            for qubit in range(self.n_qubits):
                name = f"qaoa_l{layer_idx}_q{qubit}_rz"
                layers.append(op.RZ(qubit=qubit, name=name))
                inputs[name] = 2.0 * gamma_val * h[qubit]

            for qubit in range(self.n_qubits):
                name = f"qaoa_l{layer_idx}_q{qubit}_rx"
                layers.append(op.RX(qubit=qubit, name=name))
                inputs[name] = 2.0 * beta_val

        circuit_out = _append_layers(circuit, layers, num_qubits=self.n_qubits)
        return circuit_out, inputs
