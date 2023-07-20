from dataclasses import dataclass
from typing import Dict, List, Optional

from torchbricks.bricks import BrickCollection, BrickInterface
from torchbricks.collection_utils import flatten_dict


@dataclass
class NodeEdge:
    name: str
    to_node: Optional[str]
    from_node: Optional[str]


@dataclass
class Node:
    name: str
    brick: BrickInterface
    edges: List[NodeEdge]


@dataclass
class Dag:
    nodes: Dict[str, Node]
    inputs: List[NodeEdge]
    outputs: List[NodeEdge]


def brick_collection_as_dag(brick_collections: BrickCollection) -> Dag:
    brick_collections_flat = flatten_dict(brick_collections, sep = '/')
    all_input_names = set()
    all_output_names = set()
    dag_nodes = {}
    for node_name, node in brick_collections_flat.items():
        dag_nodes[node_name] = Node(name=node_name, brick=node, edges = [])
        all_input_names = all_input_names.union(node.input_names)
        all_output_names = all_output_names.union(node.output_names)

    # Expected inputs and outputs
    only_input_names = all_input_names.difference(all_output_names)
    only_output_names = all_output_names.difference(all_input_names)

    dag_input_edges: List[NodeEdge] = []
    dag_output_edges: List[NodeEdge] = []
    for node_name, node in dag_nodes.items():
        node_input_names = set(node.brick.input_names)
        for node_only_input_name in node_input_names.intersection(only_input_names):
            dag_input_edges.append(NodeEdge(name=node_only_input_name, to_node=node_name, from_node=None))

        node_output_names = set(node.brick.output_names)
        for node_only_output_name in node_input_names.intersection(only_output_names):
            dag_output_edges.append(NodeEdge(name=node_only_output_name, to_node=None, from_node=node.name))

        # Loop all other bricks to find output_names to matching input names
        for connected_node_name, connected_node in dag_nodes.items():
            if connected_node_name == node_name:
                continue

            connected_output_names = node_output_names.intersection(connected_node.brick.input_names)
            for connected_output_name in connected_output_names:
                node.edges.append(NodeEdge(name=connected_output_name, to_node=connected_node.name, from_node=node.name))

        unconnected_outputs = only_output_names.intersection(node_output_names)
        for unconnected_output in unconnected_outputs:
            node.edges.append(NodeEdge(name=unconnected_output, to_node=None, from_node=node.name))
    return Dag(nodes=dag_nodes, inputs=dag_input_edges, outputs=dag_output_edges)
