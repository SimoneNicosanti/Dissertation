import abc

from Graph.Graph import Graph, NodeId


class ModelPartitioner(abc.ABC):

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def partition_model(self, net_node_id: NodeId, sub_graphs: list[Graph]):
        pass
