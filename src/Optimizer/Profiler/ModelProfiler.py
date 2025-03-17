import abc

import networkx as nx


class ModelProfiler(abc.ABC):

    INPUT_GENERATOR_NAME = "InputGenerator"
    OUTPUT_RECEIVER_NAME = "OutputReceiver"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def profile_model(self, input_shapes: dict[str, tuple]) -> nx.DiGraph:
        pass
