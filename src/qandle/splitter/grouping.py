import itertools
import math
import networkx as nx
from typing import Iterable, List, Set


class Node:
    def __init__(self, qubits: Iterable[int], original_index: int):
        self.qubits = tuple(sorted(dict.fromkeys(qubits)))
        self.original_index = original_index
        self.group: int = -1

        if len(self.qubits) < 2:
            raise ValueError(
                "A dependency node must operate on at least two qubits"
            )

    def __repr__(self) -> str:
        return "|".join(str(q) for q in self.qubits)

    def __hash__(self) -> int:
        return hash((self.qubits, self.original_index))

    def __lt__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (self.qubits, self.original_index) < (
            other.qubits,
            other.original_index,
        )


def groupnodes(G: nx.DiGraph, max_qubits: int) -> nx.DiGraph:
    assert max_qubits >= 2, "max_qubits must be >= 2"
    group_id = 0
    for component in nx.weakly_connected_components(G):
        subgraph = G.subgraph(component).copy()
        partitions = _partition_component(subgraph, max_qubits)
        if not partitions:
            continue
        node_to_partition = {}
        for idx, nodeset in enumerate(partitions):
            for node in nodeset:
                node_to_partition[node] = idx

        dependency_graph = nx.DiGraph()
        dependency_graph.add_nodes_from(range(len(partitions)))
        for u, v in subgraph.edges():
            gu = node_to_partition[u]
            gv = node_to_partition[v]
            if gu != gv:
                dependency_graph.add_edge(gu, gv)
        try:
            ordered = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible:
            ordered = list(range(len(partitions)))

        for idx in ordered:
            nodeset = partitions[idx]
            for node in nodeset:
                node.group = group_id
            group_id += 1
    return G


def _partition_component(subgraph: nx.DiGraph, max_qubits: int) -> List[Set[Node]]:
    undirected = subgraph.to_undirected()
    if undirected.number_of_nodes() == 0:
        return []

    component_nodes = set(undirected.nodes())
    greedy_baseline = _sequential_pack(component_nodes, max_qubits)
    baseline_count = len(greedy_baseline)

    # Use a community detection heuristic to obtain an initial partition.
    if undirected.number_of_edges() == 0:
        communities: List[Set[Node]] = [component_nodes]
    else:
        from networkx.algorithms import community

        communities = [set(c) for c in community.greedy_modularity_communities(undirected)]
        if not communities:
            communities = [component_nodes]

    partitions: List[Set[Node]] = []
    for community_nodes in communities:
        partitions.extend(
            _split_until_qubit_limit(subgraph, community_nodes, max_qubits)
        )
    if baseline_count and len(partitions) > math.ceil(1.5 * baseline_count):
        # Fall back to the greedy packing when the community heuristic produces
        # too many partitions. The greedy packing is O(n) in the number of
        # gates, which bounds the overall splitter complexity.
        return greedy_baseline
    return partitions


def _split_until_qubit_limit(
    subgraph: nx.DiGraph, nodeset: Set[Node], max_qubits: int
) -> List[Set[Node]]:
    nodes = set(nodeset)
    if not nodes:
        return []

    node_qubits = set(itertools.chain.from_iterable(n.qubits for n in nodes))
    if len(node_qubits) <= max_qubits:
        return [nodes]

    if len(nodes) == 1:
        node = next(iter(nodes))
        raise ValueError(
            f"Gate at index {node.original_index} spans {len(node.qubits)} qubits, "
            f"which exceeds the allowed maximum of {max_qubits}."
        )

    undirected = subgraph.to_undirected().subgraph(nodes).copy()
    if undirected.number_of_edges() == 0:
        return _sequential_pack(nodes, max_qubits)

    try:
        left, right = nx.algorithms.community.kernighan_lin_bisection(undirected)
    except nx.NetworkXError:
        return _sequential_pack(nodes, max_qubits)

    has_lr_edge = any(subgraph.has_edge(u, v) for u in left for v in right)
    has_rl_edge = any(subgraph.has_edge(u, v) for u in right for v in left)
    if has_lr_edge and has_rl_edge:
        return _sequential_pack(nodes, max_qubits)

    partitions: List[Set[Node]] = []
    for part in (set(left), set(right)):
        if part:
            partitions.extend(_split_until_qubit_limit(subgraph, part, max_qubits))
    return partitions


def _sequential_pack(nodes: Set[Node], max_qubits: int) -> List[Set[Node]]:
    ordered = sorted(nodes, key=lambda n: n.original_index)
    buckets: List[Set[Node]] = []
    current: List[Node] = []
    current_qubits: Set[int] = set()
    for node in ordered:
        node_qubits = set(node.qubits)
        if len(node_qubits) > max_qubits:
            raise ValueError(
                f"Gate at index {node.original_index} spans {len(node_qubits)} qubits, "
                f"which exceeds the allowed maximum of {max_qubits}."
            )
        if current and len(current_qubits | node_qubits) > max_qubits:
            buckets.append(set(current))
            current = []
            current_qubits = set()
        current.append(node)
        current_qubits |= node_qubits
    if current:
        buckets.append(set(current))
    return buckets
