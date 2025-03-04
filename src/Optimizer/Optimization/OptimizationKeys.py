from dataclasses import dataclass

from Graph.Graph import EdgeId, NodeId


@dataclass(frozen=True)
class NodeAssKey:
    mod_node_id: NodeId
    net_node_id: NodeId
    mod_name: str

    def check_model_node_and_name(self, other_node_id: NodeId, other_mod_name: str):
        return self.mod_node_id == other_node_id and self.mod_name == other_mod_name


@dataclass(frozen=True)
class EdgeAssKey:
    mod_edge_id: EdgeId
    net_edge_id: EdgeId
    mod_name: str

    def check_model_edge_and_name(self, other_edge_id: EdgeId, other_mod_name: str):
        return self.mod_edge_id == other_edge_id and self.mod_name == other_mod_name


@dataclass(frozen=True)
class MemoryUseKey:
    mod_name: str
    net_node_id: NodeId


@dataclass(frozen=True)
class ExpressionKey:
    mod_name: str
    net_node_id: NodeId
