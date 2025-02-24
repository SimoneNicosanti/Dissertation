import onnx
from Graph.AssignmentGraph import AssignmentGraph, AssignmentGraphInfo
from Graph.Graph import NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph
from Partitioner.ModelPartitioner import ModelPartitioner


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.onnx_model: onnx.ModelProto = onnx.load(self.model_path)

    def partition_model(self, assignment_graph: AssignmentGraph) -> list[str]:

        output_names = []

        for node_id in assignment_graph.get_nodes_id():
            assignment_graph_info: AssignmentGraphInfo = assignment_graph.get_node_info(
                node_id
            )
            net_node_id = assignment_graph_info.get_server_id()
            sub_graph = assignment_graph_info.get_sub_graph()
            block_idx = assignment_graph_info.get_block_idx()

            if self.__sub_graph_is_empty(sub_graph):
                continue

            input_names = self.__find_input_names(sub_graph)
            output_names = self.__find_output_names(sub_graph)

            output_path = self.model_path.replace(
                ".onnx", f"_{net_node_id}_{block_idx}.onnx"
            )
            onnx.utils.extract_model(
                self.model_path,
                output_path,
                input_names,
                output_names,
            )
        pass

    def __sub_graph_is_empty(self, sub_graph: ModelGraph) -> bool:
        if len(sub_graph.get_nodes_id()) == 1:
            if sub_graph.get_nodes_id()[0] == NodeId(
                ModelGraph.INPUT_GENERATOR_NODE_NAME
            ) or sub_graph.get_nodes_id()[0] == NodeId(
                ModelGraph.OUTPUT_RECEIVER_NODE_NAME
            ):
                return True

        return False

    def __find_input_names(self, sub_graph: ModelGraph) -> list[str]:
        input_names = set()
        for edge_id in sub_graph.get_input_edges_id():
            input_edge_info: ModelEdgeInfo = sub_graph.get_edge_info(edge_id)
            input_names = input_names.union(input_edge_info.get_tensor_names())

        return input_names

    def __find_output_names(self, sub_graph: ModelGraph) -> list[str]:
        output_names = set()
        for edge_id in sub_graph.get_output_edges_id():
            output_edge_info: ModelEdgeInfo = sub_graph.get_edge_info(edge_id)
            output_names = output_names.union(output_edge_info.get_tensor_names())

        return output_names
