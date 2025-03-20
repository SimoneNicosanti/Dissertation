import abc

import networkx as nx


from CommonProfile.NodeId import NodeId


class ModelDivider(abc.ABC):

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def divide_model(self, net_node_id: NodeId, sub_graphs: list[nx.DiGraph]):
        pass
