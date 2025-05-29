from dataclasses import dataclass

from CommonProfile.NodeId import NodeId


@dataclass(frozen=True, repr=False)
class ComponentId:
    model_name: str
    net_node_id: NodeId
    component_idx: int

    def __repr__(self):
        return f"({self.net_node_id}, {self.component_idx})"


class SolvedGraphInfo:
    SOLVED = "solved"
    VALUE = "value"
    MODEL_NAME = "model_name"

    pass


class SolvedNodeInfo:
    NET_NODE_ID = "net_node_id"
    GENERATOR = "generator"
    RECEIVER = "receiver"
    COMPONENT = "component"

    pass


class SolvedEdgeInfo:
    TENSOR_NAME_LIST = "tensor_name_list"

    pass
