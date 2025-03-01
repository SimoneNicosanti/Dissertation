import csv
import os
import tempfile

import numpy as np
import onnx
import onnx_tool
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from onnx.mapping import TENSOR_TYPE_MAP
from Profiler.ModelProfiler import ModelProfiler


class OnnxModelProfiler(ModelProfiler):

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def profile_model(self, input_shapes: dict[str, tuple]) -> ModelGraph:

        model: onnx.ModelProto = onnx.load(self.model_path)
        model_name = os.path.basename(self.model_path).removesuffix(".onnx")
        graph: ModelGraph = ModelGraph(model_name)

        self.profile_nodes(graph, model)
        self.profile_edges(graph, model)

        return graph

    def profile_nodes(self, graph: ModelGraph, model: onnx.ModelProto):

        ## Adding Fake Input Node
        graph.put_generator_node()

        ## Adding Real Nodes
        flops_dict: dict[str, float] = self.profile_nodes_flops(graph, model)
        weights_size_dict: dict[str, np.ndarray] = self.profile_node_weights_size(
            graph, model
        )
        output_size_dict: dict[str, np.ndarray] = self.profile_node_outputs_size(
            graph, model
        )

        for node in model.graph.node:
            node_id: NodeId = graph.build_node_id(node.name)
            node_info: ModelNodeInfo = ModelNodeInfo(
                model_node_flops=flops_dict[node.name],
                model_node_weights_size=weights_size_dict[node.name],
                model_node_outputs_size=output_size_dict[node.name],
            )
            graph.put_node(node_id, node_info)

        ## Adding Fake Output Node
        graph.put_receiver_node()

    def profile_nodes_flops(
        self, graph: ModelGraph, model: onnx.ModelProto
    ) -> dict[str, float]:

        m = onnx_tool.Model(m=model)

        ## TODO We are assuming static input size!!
        input_dict = {}
        for model_input in model.graph.input:
            input_shape = [1]
            for idx, dim in enumerate(model_input.type.tensor_type.shape.dim):
                if idx == 0:
                    ## Skipping batch size
                    continue
                input_shape.append(dim.dim_value)
            input_dict[model_input.name] = np.zeros(shape=tuple(input_shape))

        m.graph.shape_infer(input_dict)
        m.graph.profile()

        tempfile_name: str = tempfile.mktemp() + ".csv"
        m.graph.print_node_map(tempfile_name, metric="FLOPs")
        # m.graph.print_node_map()

        flops_dict = {}
        with open(tempfile_name, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0 or row[0] == "Total":
                    continue

                name, flops = row[0], row[2]
                flops_dict[name] = float(flops)

        return flops_dict

    def profile_node_weights_size(
        self, graph: ModelGraph, model: onnx.ModelProto
    ) -> dict[str, np.ndarray]:

        weights_size_dict: dict[str, float] = {}
        for initializer in model.graph.initializer:
            weights_size_dict[initializer.name] = self.compute_tensor_size(
                [
                    onnx.TensorShapeProto.Dimension(dim_value=d)
                    for d in initializer.dims
                ],
                initializer.data_type,
            )

        node_weights_size_dict: dict[str, float] = {}
        for node in model.graph.node:
            node_weights_size_dict.setdefault(node.name, 0)
            for input_name in node.input:
                if input_name in weights_size_dict:
                    node_weights_size_dict[node.name] += weights_size_dict[input_name]
        return node_weights_size_dict

    def profile_node_outputs_size(
        self, graph: ModelGraph, model: onnx.ModelProto
    ) -> dict[str, np.ndarray]:

        infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
            model, data_prop=True
        )

        output_size_dict: dict[str, float] = {}
        for value_info in infered_model.graph.value_info:
            output_size_dict[value_info.name] = self.compute_tensor_size(
                value_info.type.tensor_type.shape.dim,
                value_info.type.tensor_type.elem_type,
            )

        node_output_size_dict: dict[str, float] = {}
        for node in infered_model.graph.node:
            node_output_size_dict.setdefault(node.name, 0)
            for output_name in node.output:
                if output_name in output_size_dict:
                    node_output_size_dict[node.name] += output_size_dict[output_name]

        return node_output_size_dict

    def profile_edges(self, graph: ModelGraph, model: onnx.ModelProto):

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
                        tensor_data_size: float = self.compute_tensor_size(
                            tensor_info.shape.dim, tensor_info.elem_type
                        )
                        edge_id: EdgeId = graph.build_edge_id(prev_node.name, node.name)
                        edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
                        edge_info.put_tensor_name(inp_name)
                        graph.put_edge(edge_id, edge_info)

        input_graph_dict: dict[str, onnx.TypeProto.Tensor] = {
            input.name: input.type.tensor_type for input in infered_model.graph.input
        }
        for node in infered_model.graph.node:
            for inp_name in node.input:
                if inp_name in input_graph_dict.keys():
                    input_tensor = input_graph_dict[inp_name]
                    tensor_data_size: float = self.compute_tensor_size(
                        input_tensor.shape.dim, input_tensor.elem_type
                    )
                    edge_id = graph.build_edge_id(
                        ModelGraph.INPUT_GENERATOR_NODE_NAME, node.name
                    )

                    edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
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
                    tensor_data_size: float = self.compute_tensor_size(
                        output_tensor.shape.dim, output_tensor.elem_type
                    )
                    edge_id = graph.build_edge_id(
                        node.name, ModelGraph.OUTPUT_RECEIVER_NODE_NAME
                    )

                    edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
                    edge_info.put_tensor_name(out_name)
                    graph.put_edge(edge_id, edge_info)
                    graph.put_output_edge(edge_id)

    ## Tensor Size is computed in MB >> All other measures will be in MB
    def compute_tensor_size(
        self,
        tensor_shape_dim: list[onnx.TensorShapeProto.Dimension],
        tensor_elem_type: onnx.TensorProto.DataType,
    ):
        tensor_total_size = self.__init_size_in_bytes(tensor_elem_type)

        for dim in tensor_shape_dim:

            if dim.HasField("dim_param"):
                ## Batch Size
                continue
            tensor_total_size *= dim.dim_value

        return tensor_total_size / 1_000_000

    def __init_size_in_bytes(self, elem_type: onnx.TensorProto.DataType) -> int:
        map_elem = TENSOR_TYPE_MAP.get(elem_type)
        if map_elem is not None:
            ## TODO Check if this is working for integers too
            return map_elem.np_dtype.itemsize
        return 0
