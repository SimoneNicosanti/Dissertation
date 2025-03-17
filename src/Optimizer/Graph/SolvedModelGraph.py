from dataclasses import dataclass

from Optimizer.Graph.Graph import NodeId


@dataclass(frozen=True, repr=False)
class ComponentId:
    net_node_id: NodeId
    component_idx: int

    def __repr__(self):
        return f"({self.net_node_id}, {self.component_idx})"
