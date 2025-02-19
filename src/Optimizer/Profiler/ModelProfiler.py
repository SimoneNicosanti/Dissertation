import abc

from Graph.ModelGraph import ModelGraph


class ModelProfiler(abc.ABC):

    @abc.abstractmethod
    def profile_model(self, model) -> ModelGraph:
        pass
