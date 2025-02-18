import abc

from Profiler.GraphProfile import GraphProfile


class ModelProfiler(abc.ABC):

    @abc.abstractmethod
    def profile(self, model) -> GraphProfile:
        pass
