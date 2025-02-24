import csv
import tempfile

import numpy as np
import onnx
import onnx_tool
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from Profiler.ModelProfiler import ModelProfiler


class OnnxModelProfiler(ModelProfiler):

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def profile_model(self, input_shapes: dict[str, tuple]) -> ModelGraph:

        model: onnx.ModelProto = onnx.load(self.model_path)
        graph: ModelGraph = ModelGraph()

        self.init_nodes(graph, model, input_shapes)
        self.init_edges(graph, model, input_shapes)

        return graph

    def init_nodes(
        self,
        graph: ModelGraph,
        model: onnx.ModelProto,
        input_shapes: dict[str, tuple],
    ):

        m = onnx_tool.Model(m=model)

        input_dict = {}
        for inp_name, inp_shape in input_shapes.items():
            input_dict[inp_name] = np.zeros(shape=inp_shape)
        m.graph.shape_infer(input_dict)

        m.graph.profile()

        tempfile_name: str = tempfile.mktemp() + ".csv"
        m.graph.print_node_map(tempfile_name, metric="FLOPs")
        m.graph.print_node_map()

        with open(tempfile_name, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0 or row[0] == "Total":
                    continue

                name, flops = row[0], row[2]
                node_id: NodeId = NodeId(name)
                node_info = ModelNodeInfo(
                    {ModelNodeInfo.Attributes.MOD_NODE_FLOPS: float(flops)}
                )

                graph.put_node(node_id, node_info)

        ## Adding Fake Input Node
        input_node_id: NodeId = NodeId(ModelGraph.INPUT_GENERATOR_NODE_NAME)
        input_node_info: ModelNodeInfo = ModelNodeInfo(
            {ModelNodeInfo.Attributes.MOD_NODE_FLOPS: 0}
        )
        graph.put_node(input_node_id, input_node_info)

        ## Adding Fake Output Node
        output_node_id: NodeId = NodeId(ModelGraph.OUTPUT_RECEIVER_NODE_NAME)
        output_node_info: ModelNodeInfo = ModelNodeInfo(
            {ModelNodeInfo.Attributes.MOD_NODE_FLOPS: 0}
        )
        graph.put_node(output_node_id, output_node_info)

    def init_edges(
        self,
        graph: ModelGraph,
        model: onnx.ModelProto,
        input_shapes: dict[str, tuple],
    ):

        infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
            model, data_prop=True
        )

        tensor_dict: dict[str, onnx.TypeProto.Tensor] = {}
        tensor_info: onnx.ValueInfoProto

        ## Init from inference
        for tensor_info in infered_model.graph.value_info:
            tensor_dict[tensor_info.name] = tensor_info.type.tensor_type

        for node in infered_model.graph.node:
            for prev_node in infered_model.graph.node:
                for inp_name in node.input:
                    if inp_name in prev_node.output:
                        tensor_info = tensor_dict[inp_name]
                        tensor_data_size: float = self.compute_edge_data_size(
                            tensor_info
                        )
                        edge_id: EdgeId = EdgeId(
                            NodeId(prev_node.name), NodeId(node.name)
                        )
                        edge_info = ModelEdgeInfo(
                            {
                                ModelEdgeInfo.Attributes.MOD_EDGE_DATA_SIZE: tensor_data_size
                            }
                        )
                        edge_info.put_tensor_name(inp_name)
                        graph.put_edge(edge_id, edge_info)

        input_graph_dict: dict[str, onnx.TypeProto.Tensor] = {
            input.name: input.type.tensor_type for input in infered_model.graph.input
        }
        for node in infered_model.graph.node:
            for inp_name in node.input:
                if inp_name in input_graph_dict.keys():
                    input_tensor = input_graph_dict[inp_name]
                    tensor_data_size: float = self.compute_edge_data_size(input_tensor)
                    edge_id = EdgeId(
                        NodeId(ModelGraph.INPUT_GENERATOR_NODE_NAME), NodeId(node.name)
                    )

                    edge_info = ModelEdgeInfo(
                        {ModelEdgeInfo.Attributes.MOD_EDGE_DATA_SIZE: tensor_data_size}
                    )
                    edge_info.put_tensor_name(inp_name)

                    graph.put_edge(edge_id, edge_info)
                    graph.put_input_edge(edge_id)

        output_graph_dict: dict[str, onnx.TypeProto.Tensor] = {
            output.name: output.type.tensor_type
            for output in infered_model.graph.output
        }
        for node in infered_model.graph.node:
            for out_name in node.output:
                if out_name in output_graph_dict.keys():
                    output_tensor = output_graph_dict[out_name]
                    tensor_data_size: float = self.compute_edge_data_size(output_tensor)
                    edge_id = EdgeId(
                        NodeId(node.name), NodeId(ModelGraph.OUTPUT_RECEIVER_NODE_NAME)
                    )

                    edge_info = ModelEdgeInfo(
                        {ModelEdgeInfo.Attributes.MOD_EDGE_DATA_SIZE: tensor_data_size}
                    )
                    edge_info.put_tensor_name(out_name)
                    graph.put_edge(edge_id, edge_info)
                    graph.put_output_edge(edge_id)

    def compute_edge_data_size(self, tensor_info: onnx.TypeProto.Tensor):
        tensor_shape: onnx.TensorShapeProto = tensor_info.shape
        tensor_total_size = self.__init_size_in_bytes(tensor_info.elem_type)

        for dim in tensor_shape.dim:
            if dim.HasField("dim_param"):
                ## Batch Size
                continue
            tensor_total_size *= dim.dim_value

        return tensor_total_size

    def __init_size_in_bytes(self, elem_type: onnx.TensorProto.DataType) -> int:
        if elem_type == onnx.TensorProto.FLOAT:
            ## Float32 --> 4 Byte
            return 4
        elif elem_type == onnx.TensorProto.DOUBLE:
            ## Double --> 8 Byte
            return 8
        elif elem_type == onnx.TensorProto.INT8:
            ## Int8 --> 1 Byte
            return 1
