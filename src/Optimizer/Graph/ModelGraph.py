from Graph.Graph import EdgeId, Graph, GraphInfo, NodeId


class ModelNodeInfo(GraphInfo):

    def __init__(
        self,
        model_node_flops: float = 0,
        model_node_weights_size: float = 0,
        model_node_outputs_size: float = 0,
    ):
        super().__init__()
        self.model_node_flops = model_node_flops
        self.model_node_weights_size = model_node_weights_size
        self.model_node_outputs_size = model_node_outputs_size

    def get_node_flops(self) -> float:
        return self.model_node_flops

    def get_node_weights_size(self) -> float:
        return self.model_node_weights_size

    def get_node_outputs_size(self) -> float:
        return self.model_node_outputs_size


class ModelEdgeInfo(GraphInfo):

    def __init__(self, model_edge_data_size: float = 0):
        super().__init__()
        self.model_edge_data_size = model_edge_data_size
        self.tensor_names = set()

    def put_tensor_name(self, tensor_name: str):
        self.tensor_names.add(tensor_name)

    def get_tensor_names(self) -> set[str]:
        return self.tensor_names

    def get_model_edge_data_size(self) -> float:
        return self.model_edge_data_size

    def increase_data_size(self, data_size: float):
        self.model_edge_data_size += data_size


class ModelGraph(Graph):

    INPUT_GENERATOR_NODE_NAME = "InputGeneratorNode"
    OUTPUT_RECEIVER_NODE_NAME = "OutputReceiverNode"

    def __init__(self, model_name: str):
        super().__init__(model_name)

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
        edge_info_1.increase_data_size(edge_info_2.get_model_edge_data_size())
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

    def put_generator_node(self):
        node_id: NodeId = self.build_node_id(ModelGraph.INPUT_GENERATOR_NODE_NAME)
        node_info: ModelNodeInfo = ModelNodeInfo()
        self.put_node(node_id, node_info)

    def put_receiver_node(self):
        node_id: NodeId = self.build_node_id(ModelGraph.OUTPUT_RECEIVER_NODE_NAME)
        node_info: ModelNodeInfo = ModelNodeInfo()
        self.put_node(node_id, node_info)

    def is_in_graph_by_name(self, node_name: str) -> bool:
        for node_id in self.nodes.keys():
            if node_id.node_name == node_name:
                return True
        return False

    @staticmethod
    def is_generator_edge(edge_id: EdgeId) -> bool:
        return edge_id.first_node_id.node_name == ModelGraph.INPUT_GENERATOR_NODE_NAME

    @staticmethod
    def is_receiver_edge(edge_id: EdgeId) -> bool:
        return edge_id.second_node_id.node_name == ModelGraph.OUTPUT_RECEIVER_NODE_NAME

    @staticmethod
    def is_generator_node(node_id: NodeId) -> bool:
        return node_id.node_name == ModelGraph.INPUT_GENERATOR_NODE_NAME

    @staticmethod
    def is_receiver_node(node_id: NodeId) -> bool:
        return node_id.node_name == ModelGraph.OUTPUT_RECEIVER_NODE_NAME
