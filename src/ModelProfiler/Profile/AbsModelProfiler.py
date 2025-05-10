import abc

import networkx as nx


class AbsModelProfiler(abc.ABC):

    INPUT_GENERATOR_NAME = "InputGenerator"
    OUTPUT_RECEIVER_NAME = "OutputReceiver"

    @abc.abstractmethod
    def profile_model(self, model, input_shapes: dict[str, tuple]) -> nx.DiGraph:
        pass
