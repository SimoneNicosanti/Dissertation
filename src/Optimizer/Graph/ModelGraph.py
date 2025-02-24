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

    INPUT_GENERATOR_NODE_NAME = "InputGeneratorNode"
    OUTPUT_RECEIVER_NODE_NAME = "OutputReceiverNode"

    def __init__(self):
        super().__init__()

        ## Nodes generating the input
        self.input_nodes: set[NodeId] = set()
        ## Nodes receiving the output
        self.output_nodes: set[NodeId] = set()

        ## Edges transporting the input (from input node to whatever) (Even Virtual)
        self.input_edges: list[EdgeId] = []
        ## Edges transporting the output (from whatever to output node) (Even Virtual)
        self.output_edges: list[EdgeId] = []

    def __extend_model_edge_info(
        self, edge_info_1: ModelEdgeInfo, edge_info_2: ModelEdgeInfo
    ):
        edge_info_1.increase_data_size(
            edge_info_2.get_info(ModelEdgeInfo.Attributes.MOD_EDGE_DATA_SIZE)
        )
        for name in edge_info_2.get_tensor_names():
            edge_info_1.put_tensor_name(name)

    ## Extended to support additivity (Do not know if ok...)
    def put_edge(self, edge_id: EdgeId, edge_info: ModelEdgeInfo):

        if edge_id not in self.edges.keys():
            super().put_edge(edge_id, edge_info)
        else:
            curr_edge_info: ModelEdgeInfo = self.get_edge_info(edge_id)
            self.__extend_model_edge_info(curr_edge_info, edge_info)

    def get_input_nodes(self) -> list[NodeId]:
        return self.input_nodes

    def get_output_nodes(self) -> list[NodeId]:
        return self.output_nodes

    def put_input_edge(self, edge_id: EdgeId):
        self.input_edges.append(edge_id)
        self.input_nodes.add(edge_id.first_node_id)

    def get_input_edges_id(self) -> list[EdgeId]:
        return self.input_edges

    def put_output_edge(self, edge_id: EdgeId):
        self.output_edges.append(edge_id)
        self.output_nodes.add(edge_id.second_node_id)

    def get_output_edges_id(self) -> list[EdgeId]:
        return self.output_edges
