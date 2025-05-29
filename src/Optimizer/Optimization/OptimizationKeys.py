from dataclasses import dataclass

from CommonIds.NodeId import NodeId


@dataclass(frozen=True)
class NodeAssKey:
    mod_node_id: NodeId
    net_node_id: NodeId
    mod_name: str

    def check_model_node_and_name(self, other_node_id: NodeId, model_name: NodeId):
        return self.mod_node_id == other_node_id and self.mod_name == model_name


@dataclass(frozen=True)
class EdgeAssKey:
    mod_edge_id: tuple[NodeId, NodeId]
    net_edge_id: tuple[NodeId, NodeId]
    mod_name: str

    def check_model_edge_and_name(self, other_edge_id: tuple, other_mod_name: str):
        return self.mod_edge_id == other_edge_id and self.mod_name == other_mod_name


@dataclass(frozen=True)
class MemoryUseKey:
    mod_name: str
    net_node_id: NodeId
