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

        self.preprocess_model(model)

        self.init_nodes(graph, model, input_shapes)
        self.init_edges(graph, model, input_shapes)

        self.init_input_nodes(graph, model)
        self.init_output_nodes(graph, model)

        return graph

    def preprocess_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Create an identity node to wrap the input
        for input in graph.input:
            input_name = input.name
            new_input_name = input_name + "_wrapped"
            identity_node = onnx.helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[new_input_name],
                name=f"{input_name}_wrapper",
            )

            # Replace references to the input with the new wrapped version
            for node in graph.node:
                for i, inp in enumerate(node.input):
                    if inp == input_name:
                        node.input[i] = new_input_name

            # Add the new node to the graph
            graph.node.insert(0, identity_node)

        new_path = self.model_path.replace(".onnx", "_prep.onnx")
        onnx.save(model, new_path)

    def init_output_nodes(self, graph: ModelGraph, model: onnx.ModelProto):
        for node in model.graph.node:
            for mod_output in model.graph.output:
                if mod_output.name in node.output:
                    node_id = NodeId(node.name)
                    graph.put_output_node(node_id)

    def init_input_nodes(self, graph: ModelGraph, model: onnx.ModelProto):
        for node in model.graph.node:
            for mod_input in model.graph.input:
                if mod_input.name in node.input:
                    node_id = NodeId(node.name)
                    graph.put_input_node(node_id)

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

        value_info_elem: onnx.ValueInfoProto
        for value_info_elem in infered_model.graph.value_info:
            tensor_dict[value_info_elem.name] = value_info_elem.type.tensor_type

        for node in infered_model.graph.node:
            for prev_node in infered_model.graph.node:
                for inp in node.input:
                    if inp in prev_node.output:
                        tensor_shape: onnx.TensorShapeProto = tensor_dict[inp].shape
                        tensor_total_size = self.__init_size_in_bytes(
                            tensor_dict[inp].elem_type
                        )

                        for dim in tensor_shape.dim:
                            if dim.HasField("dim_param"):
                                ## Batch Size
                                continue
                            tensor_total_size *= dim.dim_value

                        edge_id: EdgeId = EdgeId(
                            NodeId(prev_node.name), NodeId(node.name)
                        )

                        if graph.get_edge_info(edge_id) is not None:
                            edge_info: ModelEdgeInfo = graph.get_edge_info(edge_id)
                            edge_info.increase_data_size(tensor_total_size)
                            edge_info.put_tensor_name(inp)
                        else:
                            edge_info = ModelEdgeInfo(
                                {
                                    ModelEdgeInfo.Attributes.MOD_EDGE_DATA_SIZE: tensor_total_size
                                }
                            )
                            edge_info.put_tensor_name(inp)
                            graph.put_edge(edge_id, edge_info)

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
