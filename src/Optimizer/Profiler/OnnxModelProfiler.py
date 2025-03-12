import csv
import os
import tempfile

import numpy as np
import onnx
import onnx_tool
from Optimizer.Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from onnx.mapping import TENSOR_TYPE_MAP
from Optimizer.Profiler.ModelProfiler import ModelProfiler


class OnnxModelProfiler(ModelProfiler):

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def profile_model(self, input_shapes: dict[str, tuple]) -> ModelGraph:

        model: onnx.ModelProto = onnx.load(self.model_path)
        model_name = os.path.basename(self.model_path).removesuffix(".onnx")
        graph: ModelGraph = ModelGraph(model_name)

        self.init_model_graph(graph, model)

        self.do_profile(graph, model)

        return graph

    def init_model_graph(
        self, model_graph: ModelGraph, model: onnx.ModelProto
    ) -> ModelGraph:

        ## Adding Fake Input Node
        model_graph.put_generator_node()

        init_nodes = []
        for input in model.graph.input:
            for node in model.graph.node:
                if input.name in node.input:
                    init_nodes.append(node)

        ## Traversing all nodes that can be reached from inputs
        traverse_queue = [node for node in init_nodes]
        visited_nodes: list[str] = []
        while traverse_queue:
            curr_node = traverse_queue.pop()
            curr_node_id = model_graph.build_node_id(curr_node.name)
            model_graph.put_node(curr_node_id, ModelNodeInfo())

            for curr_node_out in curr_node.output:
                for other_node in model.graph.node:
                    if curr_node_out in other_node.input:
                        other_node_id = model_graph.build_node_id(other_node.name)
                        model_graph.put_node(other_node_id, ModelNodeInfo())

                        edge_id = model_graph.build_edge_id(
                            curr_node.name, other_node.name
                        )
                        model_graph.put_edge(
                            edge_id,
                            ModelEdgeInfo(),
                        )

                        if other_node.name not in visited_nodes:
                            traverse_queue.append(other_node)

            visited_nodes.append(curr_node.name)

        ## Adding Fake Output Node
        model_graph.put_receiver_node()

        ## Adding Input receive edges
        for input in model.graph.input:
            for node in model.graph.node:
                if input.name in node.input:
                    edge_id = model_graph.build_edge_id(
                        ModelGraph.INPUT_GENERATOR_NODE_NAME, node.name
                    )
                    model_graph.put_edge(
                        edge_id,
                        ModelEdgeInfo(),
                    )
                    model_graph.put_input_edge(edge_id)

        ## Adding output generation edges
        for output in model.graph.output:
            for node in model.graph.node:
                if output.name in node.output:
                    edge_id = model_graph.build_edge_id(
                        node.name, ModelGraph.OUTPUT_RECEIVER_NODE_NAME
                    )
                    model_graph.put_edge(
                        edge_id,
                        ModelEdgeInfo(),
                    )
                    model_graph.put_output_edge(edge_id)

    def do_profile(self, graph: ModelGraph, onnx_model: onnx.ModelProto):

        infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
            onnx_model, data_prop=True
        )

        tensor_info_dict: dict[str, onnx.TensorProto] = {}
        for tensor_info in infered_model.graph.value_info:
            tensor_info_dict[tensor_info.name] = tensor_info

        flops_dict: dict[str, float] = self.profile_all_flops(onnx_model)

        for onnx_node in infered_model.graph.node:

            ## profile flops
            node_flops: float = flops_dict[onnx_node.name]

            ## profile weights size
            weights_size: float = self.profile_weights_size_per_node(
                onnx_node, infered_model
            )

            ## profile out edges size and total output size
            ## output_sizes_dict: dict[tuple[str, str], float] : (NextNodeName, TensorName) --> TensorSize
            output_size_dict, total_output_size = self.profile_output_size_per_node(
                onnx_node, infered_model, tensor_info_dict
            )

            ## profile if receiving input
            input_receiver_info_list: list[tuple[str, float]] = (
                self.profile_input_receive(onnx_node, infered_model)
            )

            ## profile if generating output
            output_generator_info_list: list[tuple[str, float]] = (
                self.profile_output_generation(onnx_node, infered_model)
            )

            is_reachable = graph.is_in_graph_by_name(onnx_node.name)
            if is_reachable:
                ## It is a reachable node
                self.modify_node_profile(
                    onnx_node, graph, node_flops, weights_size, total_output_size
                )
                self.modify_edge_profiles(
                    onnx_node,
                    graph,
                    output_size_dict,
                    input_receiver_info_list,
                    output_generator_info_list,
                )
            elif not is_reachable and onnx_node.op_type == "DequantizeLinear":
                ## This node is not reachable from input --> It is a node carring weights for other node!!
                next_nodes = []
                for elem in output_size_dict.keys():
                    next_nodes.append(elem[0])

                for next_node_name in next_nodes:
                    next_node_id = graph.build_node_id(next_node_name)
                    next_node_info: ModelNodeInfo = graph.get_node_info(next_node_id)
                    next_node_info.model_node_weights_size += weights_size
                pass

    def modify_node_profile(
        self,
        onnx_node: onnx.NodeProto,
        graph: ModelGraph,
        node_flops: float,
        weights_size: float,
        total_output_size: float,
    ):
        node_id = graph.build_node_id(onnx_node.name)
        node_info: ModelNodeInfo = graph.get_node_info(node_id)

        node_info.model_node_flops += node_flops
        node_info.model_node_weights_size += weights_size
        node_info.model_node_outputs_size += total_output_size

    def modify_edge_profiles(
        self,
        onnx_node: onnx.NodeProto,
        graph: ModelGraph,
        output_size_dict: dict[tuple[str, str], float],
        input_receiver_info_list: list[tuple[str, float]],
        output_generator_info_list: list[tuple[str, float]],
    ):

        next_edges: tuple[str, str] = output_size_dict.keys()

        for next_edge in next_edges:
            next_node_name, edge_tensor_name = next_edge
            edge_id = graph.build_edge_id(onnx_node.name, next_node_name)

            edge_info: ModelEdgeInfo = graph.get_edge_info(edge_id)

            edge_info.model_edge_data_size += output_size_dict[next_edge]
            edge_info.tensor_names.add(edge_tensor_name)

        for input_receiver_info in input_receiver_info_list:
            tensor_name, tensor_size = (
                input_receiver_info[0],
                input_receiver_info[1],
            )
            edge_id = graph.build_edge_id(
                ModelGraph.INPUT_GENERATOR_NODE_NAME, onnx_node.name
            )
            edge_info: ModelEdgeInfo = graph.get_edge_info(edge_id)

            edge_info.model_edge_data_size += tensor_size
            edge_info.tensor_names.add(tensor_name)

        for output_generator_info in output_generator_info_list:
            tensor_name, tensor_size = (
                output_generator_info[0],
                output_generator_info[1],
            )
            edge_id = graph.build_edge_id(
                onnx_node.name, ModelGraph.OUTPUT_RECEIVER_NODE_NAME
            )
            edge_info: ModelEdgeInfo = graph.get_edge_info(edge_id)
            edge_info.model_edge_data_size += tensor_size
            edge_info.tensor_names.add(tensor_name)

    def profile_weights_size_per_node(
        self, onnx_node: onnx.NodeProto, onnx_model: onnx.ModelProto
    ):
        total_weight_size: float = 0
        for initializer in onnx_model.graph.initializer:
            if initializer.name in onnx_node.input:
                total_weight_size += self.compute_tensor_size(
                    [
                        onnx.TensorShapeProto.Dimension(dim_value=d)
                        for d in initializer.dims
                    ],
                    initializer.data_type,
                )

        return total_weight_size

    def profile_output_size_per_node(
        self,
        onnx_node: onnx.NodeProto,
        onnx_model: onnx.ModelProto,
        tensor_info_dict: dict[str, onnx.ValueInfoProto],
    ):
        ## NextNodeName, TensorName --> TensorSize
        output_sizes_dict: dict[tuple[str, str], float] = {}
        total_output_size: float = 0

        for out_name in onnx_node.output:
            out_info: onnx.ValueInfoProto = tensor_info_dict[out_name]
            out_size: float = self.compute_tensor_size(
                out_info.type.tensor_type.shape.dim, out_info.type.tensor_type.elem_type
            )
            total_output_size += out_size

            for other_node in onnx_model.graph.node:
                if out_name in other_node.input:
                    output_sizes_dict[(other_node.name, out_name)] = out_size
        return output_sizes_dict, total_output_size

    def profile_input_receive(
        self, onnx_node: onnx.NodeProto, onnx_model: onnx.ModelProto
    ):
        received_inputs = []
        for input in onnx_model.graph.input:
            if input.name in onnx_node.input:
                input_data_size = self.compute_tensor_size(
                    input.type.tensor_type.shape.dim, input.type.tensor_type.elem_type
                )
                received_inputs.append((input.name, input_data_size))

        return received_inputs

    def profile_output_generation(
        self, onnx_node: onnx.NodeProto, onnx_model: onnx.ModelProto
    ):
        generated_outputs = []
        for output in onnx_model.graph.output:
            if output.name in onnx_node.output:
                output_data_size = self.compute_tensor_size(
                    output.type.tensor_type.shape.dim, output.type.tensor_type.elem_type
                )
                generated_outputs.append((output.name, output_data_size))

        return generated_outputs

    def profile_all_flops(self, model: onnx.ModelProto) -> dict[str, float]:

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

        flops_dict = {}
        with open(tempfile_name, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0 or row[0] == "Total":
                    continue

                name, flops = row[0], row[2]
                flops_dict[name] = float(flops)

        return flops_dict

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

        ## Tensor Size is computed in MB >> All other measures will be in MB
        return tensor_total_size / 1_000_000

    def __init_size_in_bytes(self, elem_type: onnx.TensorProto.DataType) -> int:
        map_elem = TENSOR_TYPE_MAP.get(elem_type)
        if map_elem is not None:
            ## TODO Check if this is working for integers too
            return map_elem.np_dtype.itemsize
        return 0
