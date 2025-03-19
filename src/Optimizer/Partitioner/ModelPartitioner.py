import abc

import networkx as nx

from Optimizer.Graph.Graph import NodeId


class ModelPartitioner(abc.ABC):

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def partition_model(self, net_node_id: NodeId, sub_graphs: list[nx.DiGraph]):
        pass
