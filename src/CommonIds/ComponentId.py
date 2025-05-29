import ast
from dataclasses import dataclass

from CommonIds.NodeId import NodeId


@dataclass(frozen=True, repr=False)
class ComponentId:
    model_name: str
    net_node_id: NodeId
    component_idx: int

    def __repr__(self):
        return f"({self.net_node_id}, {self.component_idx})"

    def encode(self):
        return str((self.model_name, self.net_node_id.node_name, self.component_idx))

    @staticmethod
    def decode(encoded_component_id: str):
        component_id_tuple = ast.literal_eval(encoded_component_id)
        return ComponentId(
            model_name=component_id_tuple[0],
            net_node_id=NodeId(node_name=component_id_tuple[1]),
            component_idx=component_id_tuple[2],
        )
