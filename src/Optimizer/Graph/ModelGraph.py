import abc
from enum import Enum

from Graph.Graph import EdgeId, Graph, GraphInfo, NodeId


class ModelNodeInfo(GraphInfo):

    class Attributes(Enum):
        MOD_NODE_FLOPS = 0

    def __init__(self, info_dict: dict[str, float]):
        super().__init__(info_dict)


class ModelEdgeInfo(GraphInfo):

    class Attributes(Enum):
        MOD_EDGE_DATA_SIZE = 0

    def __init__(self, info_dict: dict[str, float]):
        super().__init__(info_dict)
        self.tensor_names = set()

    def put_tensor_name(self, tensor_name: str):
        self.tensor_names.add(tensor_name)

    def get_tensor_names(self) -> set[str]:
        return self.tensor_names

    def increase_data_size(self, data_size: float):
        self.info_dict[self.Attributes.MOD_EDGE_DATA_SIZE.value] += data_size


class ModelGraph(Graph, abc.ABC):

    def __init__(self):
        super().__init__()
        self.input_nodes: list[NodeId] = []
        self.output_nodes: list[NodeId] = []

        self.input_edges: dict[EdgeId, ModelEdgeInfo] = {}
        self.output_edges: list[EdgeId, ModelEdgeInfo] = {}

        pass

    def put_input_node(self, node: NodeId):
        self.input_nodes.append(node)

    def get_input_nodes(self):
        return self.input_nodes

    def put_output_node(self, node: NodeId):
        self.output_nodes.append(node)

    def get_output_nodes(self):
        return self.output_nodes

    def put_input_edge(self, edge: EdgeId, edge_info: ModelEdgeInfo):
        self.input_edges[edge] = edge_info

    def get_input_edges_id(self) -> list[EdgeId]:
        return list(self.input_edges.keys())

    def get_input_edge_info(self, edge_id: EdgeId) -> ModelEdgeInfo:
        return self.input_edges[edge_id]

    def put_output_edge(self, edge: EdgeId, edge_info: ModelEdgeInfo):
        self.output_edges[edge] = edge_info

    def get_output_edges_id(self) -> list[EdgeId]:
        return list(self.output_edges.keys())

    def get_output_edge_info(self, edge_id: EdgeId) -> ModelEdgeInfo:
        return self.output_edges[edge_id]
