
from typing import Dict, Union

from torchbricks.bricks import BrickCollection
from torchbricks.collection_utils import unflatten
from torchbricks.dag import Dag, Node, brick_collection_as_dag


def create_mermaid_dag_graph(brick_collection: BrickCollection, add_legends: bool = True) -> str:
    dag = brick_collection_as_dag(brick_collection)
    dag_builder = MermaidDagBuilder()
    dag_builder.start_graph()
    dag_builder.add_comment("Brick definitions")
    for node in dag.nodes.values():
        dag_builder.define_node(node)

    dag_builder.add_empty_line()
    dag_builder.add_comment("Draw input and outputs")
    for dag_input in dag.inputs:
        string = f"{dag_input.name}:::input --> {dag_input.to_node}"
        dag_builder.add_string(string=string)

    dag_builder.add_empty_line()
    dag_builder.add_comment("Draw nodes and edges")
    dag_builder.draw_nodes_and_connections(dag_nodes=dag.nodes)


    dag_builder.add_empty_line()
    dag_builder.add_comment("Add styling")
    dag_builder.add_style_from_string(style_name="arrow", style="stroke-width:0px,fill-opacity:0.0", add_to_legends=False)
    dag_builder.add_style_from_string(style_name="input", style="stroke-width:0px,fill-opacity:0.3,fill:#22A699")
    dag_builder.add_style_from_string(style_name="output", style="stroke-width:0px,fill-opacity:0.3,fill:#F2BE22")
    dag_builder.add_node_styles(dag=dag)

    if add_legends:
        dag_builder.add_empty_line()
        dag_builder.add_comment("Add legends")
        dag_builder.add_legends()
    return dag_builder.build()


class MermaidDagBuilder:  # Pipeline pattern
    def __init__(self) -> None:
        self._graph_lines = []
        self._indent_level = 0
        self._indent_char = "    "
        self._added_legends = []

    def add_string(self, string: str, increase_indent_level: bool = False, decrease_indent_level: bool = False):
        if decrease_indent_level:
            self._indent_level = self._indent_level-1

        indent_chars = self._indent_char*self._indent_level
        self._graph_lines.append(indent_chars + string)
        if increase_indent_level:
            self._indent_level = self._indent_level+1

    def add_empty_line(self):
        self.add_string(string="")

    def add_comment(self, string: str):
        self.add_string(f"%% {string}")

    def start_graph(self):
        self.add_string("flowchart LR", increase_indent_level=True)

    def define_node(self, brick: Node):
        module_name = brick.brick.get_module_name()
        style = brick.brick.get_brick_type()
        brick_string = f"{brick.name}(<strong>'{brick.name}': {module_name}</strong><br><i>{style}</i>):::{style}"
        self.add_string(brick_string)

    def draw_nodes_and_connections(self, dag_nodes: Dict[str, Node]):
        nodes_nested = unflatten(dag_nodes)
        self._draw_nested_nodes_recursive(nodes_nested)

    def build(self, markdown=True) -> str:
        graph_lines = list(self._graph_lines)
        if markdown:
            graph_lines.insert(0, "```mermaid")
            graph_lines.append("```")
        mermaid_str = "\n".join(graph_lines)
        return mermaid_str

    def _draw_nested_nodes_recursive(self, nodes_nested: Dict[str, Union[Node, Dict]]):
        for node_or_subgraph_name, node_or_subgraph in nodes_nested.items():
            if isinstance(node_or_subgraph, dict):
                subgraph = node_or_subgraph
                self.add_string(f"subgraph {node_or_subgraph_name}", increase_indent_level=True)
                self._draw_nested_nodes_recursive(subgraph)
                self.add_string("end", decrease_indent_level=True)
            else:
                node: Node = node_or_subgraph
                for brick_connection in node.edges:
                    if brick_connection.to_node is None:
                        string = f"{node.name} --> {brick_connection.name}:::output"
                    else:
                        string = f"{node.name} --> |{brick_connection.name}| {brick_connection.to_node}"
                    self.add_string(string=string)

    def add_style_from_string(self, style: str, style_name: str, add_to_legends: bool = True ):
        if add_to_legends:
            self._added_legends.append(style_name)
        self.add_string(f"classDef {style_name} {style} ")

    def add_style_from_dict(self, style: Dict[str, str], style_name: str, add_to_legends: bool = True):
        html_style = ",".join([f"{style_attr_name}:{style_attr_value}" for style_attr_name, style_attr_value in style.items()])
        self.add_style_from_string(style=html_style, style_name=style_name, add_to_legends=add_to_legends)

    def add_node_styles(self, dag: Dag, add_to_legends: bool = False):
        node_styles = set()
        for node in dag.nodes.values():
            style_name = node.brick.get_brick_type()
            if style_name in node_styles:
                continue
            node_styles.add(style_name)
            style_name = node.brick.get_brick_type()
            self.add_style_from_dict(style=node.brick.style, style_name=style_name, add_to_legends=add_to_legends)

    def add_legends(self):
        self.add_string("subgraph Legends", increase_indent_level=True)
        for style in self._added_legends:
            self.add_string(f"{style}({style}):::{style}")
        self.add_string("end", decrease_indent_level=True)
