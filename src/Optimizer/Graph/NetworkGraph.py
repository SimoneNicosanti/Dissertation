from enum import Enum

from Graph.Graph import Graph, GraphInfo


class NetworkGraph(Graph):

    def __init__(self):
        super().__init__()


class NetworkEdgeInfo(GraphInfo):

    class Attributes(Enum):
        NET_EDGE_BANDWIDTH = 0

    def __init__(self, info_dict: dict[str, float]):
        super().__init__(info_dict)


class NetworkNodeInfo(GraphInfo):

    class Attributes(Enum):
        NET_NODE_FLOPS_PER_SEC = 0
        NET_NODE_COMP_ENERGY_PER_SEC = 1
        NET_NODE_TRANS_ENERGY_PER_SEC = 2

    def __init__(self, info_dict: dict[str, float]):
        super().__init__(info_dict)
