import abc

from Graph.Graph import Graph


class ModelPartitioner(abc.ABC):

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def partition_model(self, sub_graphs: list[Graph]) -> list[str]:
        pass
