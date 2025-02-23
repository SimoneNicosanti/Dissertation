import onnx
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from Partitioner.ModelPartitioner import ModelPartitioner


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.onnx_model: onnx.ModelProto = onnx.load(self.model_path)

    def partition_model(
        self, net_node_id: NodeId, sub_graphs: list[ModelGraph]
    ) -> list[str]:
        for idx, sub_graph in enumerate(sub_graphs):

            input_names = self.__find_input_names(sub_graph)
            output_names = self.__find_output_names(sub_graph)

            output_path = self.model_path.replace(".onnx", f"_sub_{idx}.onnx")
            onnx.utils.extract_model(
                self.model_path,
                output_path,
                input_names,
                output_names,
            )
        pass

    def __find_input_names(self, sub_graph: ModelGraph) -> list[str]:
        input_names = set()

        for edge_id in sub_graph.get_input_edges_id():
            input_edge_info: ModelEdgeInfo = sub_graph.get_input_edge_info(edge_id)
            input_names = input_names.union(input_edge_info.get_tensor_names())

        print(input_names)
        return input_names

    def __find_output_names(self, sub_graph: ModelGraph) -> list[str]:
        output_names = set()

        for edge_id in sub_graph.get_output_edges_id():
            output_edge_info: ModelEdgeInfo = sub_graph.get_output_edge_info(edge_id)
            output_names = output_names.union(output_edge_info.get_tensor_names())

        print("Output >> ", output_names)

        return output_names
