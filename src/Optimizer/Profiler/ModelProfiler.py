import abc

from Optimizer.Graph.ModelGraph import ModelGraph


class ModelProfiler(abc.ABC):

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def profile_model(self, input_shapes: dict[str, tuple]) -> ModelGraph:
        pass
