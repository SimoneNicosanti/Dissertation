from Graph.Graph import Edge, Graph, Node
from Profiler.GraphProfile import GraphProfile


class ModelGraph(Graph):

    def __init__(self, model_profile: GraphProfile):
        self.model_nodes: list[Node] = []
        self.model_edges: list[Edge] = []

        self.model_profile: GraphProfile = model_profile

        pass

    def get_nodes(self) -> list[Node]:
        return self.model_nodes

    def get_edges(self) -> list[Edge]:
        return self.model_edges
