import math
import typing
import networkx as nx
import dataclasses
import qandle
import qandle.splitter.grouping as grouping
import qandle.operators as op

cnot_types = typing.Union[op.CNOT, op.BuiltCNOT]
cz_types = typing.Union[op.CZ, op.BuiltCZ]
swap_types = typing.Union[op.SWAP, op.BuiltSWAP]
ccnot_types = typing.Union[op.CCNOT, op.BuiltCCNOT]
controlled_types = typing.Union[op.Controlled, op.BuiltControlled]

def split(
    circuit,
    max_qubits: int = 5,
) -> typing.Dict[int, "SubcircuitContainer"]:
    assert circuit.num_qubits > 1, "Can't split circuits with only 1 qubit"
    assert all(
        isinstance(layer, op.Operator) for layer in circuit.layers
    ), f"Unknown layer type in circuit: {[type(layer) for layer in circuit.layers if not isinstance(layer, op.Operator)]}"
    layers = list(circuit.layers)
    temp_circuit = qandle.Circuit(num_qubits=circuit.num_qubits, layers=layers)
    G = _construct_graph(temp_circuit)
    isolated_qubits = G.graph.get("unused_qubits", set())
    G = grouping.groupnodes(G, max_qubits)
    assigned = _assign_to_subcircuits(
        G.nodes(), layers, circuit.num_qubits, isolated_qubits, max_qubits
    )
    normalized = _normalize_subcircuits(assigned)
    return normalized

def _construct_graph(circuit) -> nx.DiGraph:
    circuit = circuit.decompose().circuit
    nodes = []
    for i, layer in enumerate(circuit.layers):
        if _is_multi_qubit_gate(layer):
            qubits = _extract_gate_qubits(layer)
            if len(qubits) >= 2:
                nodes.append(grouping.Node(qubits, i))

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)

    used_qubits = set(q for node in nodes for q in node.qubits)
    unused_qubits = set(range(circuit.num_qubits)) - used_qubits
    G.graph["unused_qubits"] = unused_qubits
    if not nodes:
        return G

    nodes_per_qubit = {w: [] for w in range(circuit.num_qubits)}
    for n in nodes:
        for q in n.qubits:
            nodes_per_qubit[q].append(n)

    for w in range(circuit.num_qubits):
        for idx in range(len(nodes_per_qubit[w]) - 1):
            G.add_edge(nodes_per_qubit[w][idx], nodes_per_qubit[w][idx + 1])
    return G


def _assign_to_subcircuits(
    nodes: typing.Iterable[grouping.Node],
    original_circuit_layers: list,
    num_qubits: int,
    isolated_qubits: typing.Iterable[int],
    max_qubits: int,
) -> typing.Dict[int, typing.List[op.Operator]]:
    """
    From the list of nodes (which have their new group assigned), sort the original layers into subcircuits
    """

    node_list = list(nodes)
    if not node_list:
        if not original_circuit_layers:
            return {}
        # No entangling operations: keep the circuit as a single subcircuit.
        return {0: list(original_circuit_layers)}

    groups = sorted({n.group for n in node_list})
    subcircuits = {group: dict() for group in groups}
    node_per_index = {n.original_index: n for n in node_list}

    qubits_to_nodes: typing.Dict[int, typing.List[grouping.Node]] = {
        q: [] for q in range(num_qubits)
    }
    for n in node_list:
        for q in n.qubits:
            qubits_to_nodes[q].append(n)
    for qubits_nodes in qubits_to_nodes.values():
        qubits_nodes.sort(key=lambda n: n.original_index)

    isolated_set = set(isolated_qubits)
    isolated_ops: typing.Dict[int, typing.List[typing.Tuple[int, op.Operator]]] = {
        q: [] for q in sorted(isolated_set)
    }

    def _closest_group(layer_index: int, qubits: typing.Set[int]) -> typing.Optional[int]:
        candidates: typing.List[grouping.Node] = []
        for q in qubits:
            candidates.extend(qubits_to_nodes.get(q, []))
        if not candidates:
            return None
        closest = min(
            candidates,
            key=lambda n: (abs(n.original_index - layer_index), n.original_index),
        )
        return closest.group

    for i, layer in enumerate(original_circuit_layers):
        qubits = set(_extract_gate_qubits(layer))
        if not qubits:
            # Assign global operations to the earliest group to preserve order.
            subcircuits[groups[0]][i] = layer
            continue
        if _is_multi_qubit_gate(layer):
            n = node_per_index[i]
            subcircuits[n.group][i] = layer
            continue

        target_group = _closest_group(i, qubits)
        if target_group is not None:
            subcircuits[target_group][i] = layer
            continue

        if qubits <= isolated_set:
            # Qubits never entangle: keep their operations together.
            q = next(iter(qubits))
            isolated_ops[q].append((i, layer))
            continue

        # Fallback: place the gate in the earliest subcircuit. This path should
        # be extremely rare but keeps the splitter robust even when metadata is missing.
        subcircuits[groups[0]][i] = layer

    next_group = (max(groups) + 1) if groups else 0
    offset = len(original_circuit_layers)
    for q in sorted(isolated_ops.keys()):
        ops = isolated_ops[q]
        if not ops:
            continue
        subcircuits[next_group] = {
            offset + idx: layer for idx, (original_idx, layer) in enumerate(ops)
        }
        next_group += 1

    # flatten the subcircuits while preserving global gate ordering. When the same
    # group appears in disjoint time windows we emit multiple subcircuits to keep
    # execution chronological.
    ordered_ops = sorted(
        (
            (idx, group, layer)
            for group, layers in subcircuits.items()
            for idx, layer in layers.items()
        ),
        key=lambda triple: triple[0],
    )
    flat_subc: dict[int, typing.List[op.Operator]] = {}
    current_group: typing.Any = None
    current_layers: typing.List[op.Operator] = []
    next_id = 0
    for _, group, layer in ordered_ops:
        if current_group is None or group != current_group:
            if current_layers:
                flat_subc[next_id] = list(current_layers)
                next_id += 1
            current_group = group
            current_layers = [layer]
        else:
            current_layers.append(layer)
    if current_layers:
        flat_subc[next_id] = list(current_layers)

    if ordered_ops:
        greedy_segments: typing.List[typing.List[op.Operator]] = []
        current_layers = []
        current_qubits: typing.Set[int] = set()
        for _, _, layer in ordered_ops:
            layer_qubits = set(_extract_gate_qubits(layer))
            if len(layer_qubits) > max_qubits:
                raise ValueError(
                    f"Gate {layer} spans {len(layer_qubits)} qubits, exceeding the limit of {max_qubits}."
                )
            if current_layers and len(current_qubits | layer_qubits) > max_qubits:
                greedy_segments.append(list(current_layers))
                current_layers = []
                current_qubits = set()
            current_layers.append(layer)
            current_qubits |= layer_qubits
        if current_layers:
            greedy_segments.append(list(current_layers))
        allowed_segments = math.ceil(1.5 * len(greedy_segments)) if greedy_segments else 0
        if greedy_segments and len(flat_subc) > allowed_segments:
            flat_subc = {i: segment for i, segment in enumerate(greedy_segments)}
    return flat_subc


@dataclasses.dataclass
class SubcircuitContainer:
    layers: list  # list of gates
    mapping: dict  # original index -> normalised index


def _normalize_subcircuits(assigned: dict):
    """
    subcircuits may have gaps in the qubit indices, remove those e.g.
    [CNOT(0,1), CNOT(4,5)] becomes [CNOT(0,1), CNOT(2,3)].
    Also returns a mapping from the original index to the normalised index
    (so {0:0, 1:1, 4:2, 5:3} in the example above)
    """

    def normalize_subcircuit(subcircuit: list) -> SubcircuitContainer:
        all_qubits = sorted(
            set(q for gate in subcircuit for q in _extract_gate_qubits(gate))
        )
        mapping = {q: i for i, q in enumerate(all_qubits)}
        new_layers = []
        for g in subcircuit:
            if isinstance(g, cnot_types):
                new_layers.append(
                    op.CNOT(control=mapping[g.c], target=mapping[g.t])
                )
            elif isinstance(g, cz_types):
                new_layers.append(
                    op.CZ(control=mapping[g.c], target=mapping[g.t])
                )
            elif isinstance(g, swap_types):
                if hasattr(g, "a") and hasattr(g, "b"):
                    new_layers.append(
                        op.SWAP(a=mapping[g.a], b=mapping[g.b])
                    )
                else:
                    new_layers.append(
                        op.SWAP(a=mapping[g.c], b=mapping[g.t])
                    )
            elif isinstance(g, op.CustomGate):
                gnew = op.CustomGate(
                    qubit=mapping[g.qubit],
                    matrix=g.original_matrix,
                    num_qubits=len(all_qubits),
                    self_description=g.description,
                )
                new_layers.append(gnew)
            elif isinstance(g, op.U):
                new_layers.append(
                    op.U(
                        qubit=mapping[g.qubit],
                        matrix=g.matrix,
                    )
                )
            else:
                c = (
                    op.BUILT_CLASS_RELATION.T[type(g)]
                    if isinstance(g, op.BuiltOperator)
                    else type(g)
                )
                otherargs = {}
                if hasattr(g, "theta") and g.theta is not None:
                    otherargs["theta"] = g.theta
                if hasattr(g, "name") and g.name is not None:
                    otherargs["name"] = g.name
                if hasattr(g, "remapping"):
                    otherargs["remapping"] = g.remapping

                if hasattr(g, "qubit"):
                    new_layers.append(c(qubit=mapping[g.qubit], **otherargs))
                elif hasattr(g, "qubits"):
                    otherargs["qubits"] = [mapping[q] for q in g.qubits]
                    new_layers.append(c(**otherargs))
                elif hasattr(g, "a") and hasattr(g, "b"):
                    new_layers.append(
                        c(a=mapping[g.a], b=mapping[g.b], **otherargs)
                    )
                elif (
                    isinstance(g, controlled_types)
                    or (
                        hasattr(g, "c")
                        and hasattr(g, "t")
                        and not isinstance(g, cnot_types)
                    )
                ):
                    if isinstance(g, controlled_types):
                        target_gate = _remap_controlled_target(
                            g.t, mapping, len(all_qubits)
                        )
                        new_gate = op.Controlled(
                            control=mapping[g.c], target=target_gate
                        )
                        if isinstance(g, op.BuiltControlled):
                            new_gate = new_gate.build(len(all_qubits))
                        else:
                            new_gate = new_gate
                        new_layers.append(new_gate)
                    else:
                        new_layers.append(
                            c(control=mapping[g.c], target=mapping[g.t], **otherargs)
                        )
                elif (
                    hasattr(g, "c1")
                    and hasattr(g, "c2")
                    and hasattr(g, "t")
                ):
                    new_layers.append(
                        c(
                            control1=mapping[g.c1],
                            control2=mapping[g.c2],
                            target=mapping[g.t],
                            **otherargs,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported gate type {type(g)} in splitter normalization"
                    )
        return SubcircuitContainer(new_layers, mapping)

    normalized = {i: normalize_subcircuit(assigned[i]) for i in assigned.keys()}
    return normalized


def _extract_gate_qubits(gate, _visited=None) -> typing.Tuple[int, ...]:
    if _visited is None:
        _visited = set()
    obj_id = id(gate)
    if obj_id in _visited:
        return tuple()
    _visited.add(obj_id)

    qubits: typing.List[int] = []

    if hasattr(gate, "qubits"):
        qubits_attr = getattr(gate, "qubits")
        qubits.extend(_collect_qubits_from_value(qubits_attr, _visited))

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
            qubits.extend(_collect_qubits_from_value(value, _visited))

    unique_qubits = tuple(sorted(dict.fromkeys(q for q in qubits if isinstance(q, int))))
    return unique_qubits


def _collect_qubits_from_value(value, visited) -> typing.List[int]:
    collected: typing.List[int] = []
    if isinstance(value, int):
        collected.append(value)
    elif isinstance(value, (list, tuple, set)):
        for elem in value:
            collected.extend(_collect_qubits_from_value(elem, visited))
    elif isinstance(value, dict):
        for elem in value.values():
            collected.extend(_collect_qubits_from_value(elem, visited))
    elif hasattr(value, "qubit") or hasattr(value, "qubits"):
        collected.extend(_extract_gate_qubits(value, visited))
    return collected


def _is_multi_qubit_gate(gate) -> bool:
    if isinstance(gate, (cnot_types, cz_types, swap_types, ccnot_types, controlled_types)):
        return True
    qubits = _extract_gate_qubits(gate)
    return len(qubits) >= 2


def _remap_controlled_target(gate, mapping: dict, num_qubits: int):
    if isinstance(gate, op.BuiltOperator):
        target_cls = op.BUILT_CLASS_RELATION.T.get(type(gate))
        if target_cls is None:
            raise ValueError(f"Unsupported controlled target type {type(gate)}")
        kwargs = {}
        if hasattr(gate, "theta") and gate.theta is not None:
            kwargs["theta"] = gate.theta
        if hasattr(gate, "name") and gate.name is not None:
            kwargs["name"] = gate.name
        if hasattr(gate, "remapping"):
            kwargs["remapping"] = gate.remapping
        if hasattr(gate, "original_matrix"):
            kwargs["matrix"] = gate.original_matrix
        if hasattr(gate, "matrix") and "matrix" not in kwargs:
            kwargs["matrix"] = gate.matrix
        if hasattr(gate, "qubit"):
            kwargs["qubit"] = mapping[gate.qubit]
        elif hasattr(gate, "qubits"):
            kwargs["qubits"] = [mapping[q] for q in gate.qubits]
        elif hasattr(gate, "a") and hasattr(gate, "b"):
            kwargs["a"] = mapping[gate.a]
            kwargs["b"] = mapping[gate.b]
        elif hasattr(gate, "c") and hasattr(gate, "t"):
            kwargs["control"] = mapping[gate.c]
            kwargs["target"] = mapping[gate.t]
        elif hasattr(gate, "c1") and hasattr(gate, "c2") and hasattr(gate, "t"):
            kwargs["control1"] = mapping[gate.c1]
            kwargs["control2"] = mapping[gate.c2]
            kwargs["target"] = mapping[gate.t]
        else:
            raise ValueError(
                f"Unsupported attributes for controlled target remapping: {gate}"
            )
        unbuilt = target_cls(**kwargs)
    else:
        target_cls = type(gate)
        kwargs = {}
        if hasattr(gate, "theta") and gate.theta is not None:
            kwargs["theta"] = gate.theta
        if hasattr(gate, "name") and gate.name is not None:
            kwargs["name"] = gate.name
        if hasattr(gate, "remapping"):
            kwargs["remapping"] = gate.remapping
        if hasattr(gate, "matrix"):
            kwargs["matrix"] = gate.matrix
        if hasattr(gate, "qubit"):
            kwargs["qubit"] = mapping[gate.qubit]
        elif hasattr(gate, "qubits"):
            kwargs["qubits"] = [mapping[q] for q in gate.qubits]
        elif hasattr(gate, "a") and hasattr(gate, "b"):
            kwargs["a"] = mapping[gate.a]
            kwargs["b"] = mapping[gate.b]
        elif hasattr(gate, "c") and hasattr(gate, "t"):
            kwargs["control"] = mapping[gate.c]
            kwargs["target"] = mapping[gate.t]
        elif hasattr(gate, "c1") and hasattr(gate, "c2") and hasattr(gate, "t"):
            kwargs["control1"] = mapping[gate.c1]
            kwargs["control2"] = mapping[gate.c2]
            kwargs["target"] = mapping[gate.t]
        else:
            raise ValueError(
                f"Unsupported attributes for controlled target remapping: {gate}"
            )
        unbuilt = target_cls(**kwargs)
    # Ensure we return unbuilt operator to allow Controlled.build to adapt dimensions
    if hasattr(unbuilt, "build"):
        return unbuilt
    return unbuilt
