import abc

from GraphId import EdgeId, NodeId
from Profiler.GraphProfile import (
    EdgeInfo,
    NodeInfo,
)


class Edge:

    def __init__(self, edge_id: EdgeId, edge_info: EdgeInfo):
        self.edge_id = edge_id
        self.edge_info = edge_info

    def get_edge_info(self) -> EdgeInfo:
        return self.edge_info

    def get_edge_id(self) -> EdgeId:
        return self.edge_id


class Node:

    def __init__(self, node_id: NodeId, node_info: NodeInfo):
        self.node_id = node_id
        self.node_info: NodeInfo = node_info
        pass

    def get_node_info(self) -> NodeInfo:
        return self.node_info

    def get_node_id(self) -> NodeId:
        return self.node_id


class Graph(abc.ABC):

    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

        pass

    @abc.abstractmethod
    def init_graph(self) -> None:
        pass

    def get_nodes(self) -> list[Node]:
        return self.nodes

    def get_edges(self) -> list[Edge]:
        return self.edges
