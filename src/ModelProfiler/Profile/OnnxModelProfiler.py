import csv
import os
import tempfile

import networkx as nx
import numpy as np
import onnx
import onnx_tool
from onnx.mapping import TENSOR_TYPE_MAP

from CommonProfile.ModelInfo import ModelEdgeInfo, ModelNodeInfo
from CommonProfile.NodeId import NodeId
from ModelProfiler.Profile.AbsModelProfiler import AbsModelProfiler


class OnnxModelProfiler(AbsModelProfiler):

    def __init__(self) -> None:
        super().__init__()

    def profile_model(
        self, model: onnx.ModelProto, model_name: str, input_shapes: dict[str, tuple]
    ) -> nx.DiGraph:

        graph: nx.DiGraph = nx.DiGraph(name=model_name)

        self.init_model_graph(graph, model)
        print("Done Init")

        self.do_profile(graph, model)
        print("Done Profile")

        return graph

    def init_model_graph(self, model_graph: nx.DiGraph, model: onnx.ModelProto):

        ## Adding Fake Input Node
        node_idx = 0
        model_graph.add_node(
            NodeId(AbsModelProfiler.INPUT_GENERATOR_NAME),
            idx=node_idx,
            flops=0,
            weights_size=0,
            outputs_size=0,
            generator=True,
        )

        node_idx += 1

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
            node_id = NodeId(curr_node.name)
            if not model_graph.has_node(node_id):
                model_graph.add_node(node_id, idx=node_idx)
                node_idx += 1

            for curr_node_out in curr_node.output:
                for other_node in model.graph.node:
                    if curr_node_out in other_node.input:

                        other_node_id = NodeId(other_node.name)
                        if not model_graph.has_node(other_node_id):
                            model_graph.add_node(other_node_id, idx=node_idx)
                            node_idx += 1

                        model_graph.add_edge(node_id, other_node_id)

                        if other_node.name not in visited_nodes:
                            traverse_queue.append(other_node)

            visited_nodes.append(curr_node.name)

        ## Adding Fake Output Node
        model_graph.add_node(
            NodeId(AbsModelProfiler.OUTPUT_RECEIVER_NAME),
            idx=node_idx,
            receiver=True,
            flops=0,
            weights_size=0,
            outputs_size=0,
        )
        node_idx += 1

        ## Adding Input receive edges
        for input in model.graph.input:
            for node in model.graph.node:
                if input.name in node.input:
                    model_graph.add_edge(
                        NodeId(AbsModelProfiler.INPUT_GENERATOR_NAME),
                        NodeId(node.name),
                    )

        ## Adding output generation edges
        for output in model.graph.output:
            for node in model.graph.node:
                if output.name in node.output:
                    model_graph.add_edge(
                        NodeId(node.name), NodeId(AbsModelProfiler.OUTPUT_RECEIVER_NAME)
                    )

    def do_profile(self, graph: nx.DiGraph, onnx_model: onnx.ModelProto):

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

            is_reachable = graph.has_node(NodeId(onnx_node.name))
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
                    next_node_id = NodeId(next_node_name)
                    graph.nodes[next_node_id].setdefault(ModelNodeInfo.WEIGHTS_SIZE, 0)
                    graph.nodes[next_node_id][
                        ModelNodeInfo.WEIGHTS_SIZE
                    ] += weights_size
                pass

    def modify_node_profile(
        self,
        onnx_node: onnx.NodeProto,
        graph: nx.DiGraph,
        node_flops: float,
        weights_size: float,
        total_output_size: float,
    ):
        node_id = NodeId(onnx_node.name)
        graph.nodes[node_id].setdefault(ModelNodeInfo.FLOPS, 0)
        graph.nodes[node_id].setdefault(ModelNodeInfo.WEIGHTS_SIZE, 0)
        graph.nodes[node_id].setdefault(ModelNodeInfo.OUTPUTS_SIZE, 0)

        graph.nodes[node_id][ModelNodeInfo.FLOPS] += node_flops
        graph.nodes[node_id][ModelNodeInfo.WEIGHTS_SIZE] += weights_size
        graph.nodes[node_id][ModelNodeInfo.OUTPUTS_SIZE] += total_output_size

    def modify_edge_profiles(
        self,
        onnx_node: onnx.NodeProto,
        graph: nx.DiGraph,
        output_size_dict: dict[tuple[str, str], float],
        input_receiver_info_list: list[tuple[str, float]],
        output_generator_info_list: list[tuple[str, float]],
    ):
        node_id = NodeId(onnx_node.name)
        next_edges: tuple[str, str] = output_size_dict.keys()

        for next_edge in next_edges:
            next_node_name, tensor_name = next_edge
            next_node_id = NodeId(next_node_name)

            edge_id = (node_id, next_node_id)

            graph.edges[edge_id].setdefault(ModelEdgeInfo.TOT_TENSOR_SIZE, 0)
            graph.edges[edge_id].setdefault(ModelEdgeInfo.TENSOR_NAME_LIST, [])

            graph.edges[edge_id][ModelEdgeInfo.TOT_TENSOR_SIZE] += output_size_dict[
                next_edge
            ]

            if tensor_name not in graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST]:
                graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST].append(tensor_name)

        for input_receiver_info in input_receiver_info_list:

            edge_id = (
                NodeId(AbsModelProfiler.INPUT_GENERATOR_NAME),
                NodeId(onnx_node.name),
            )
            graph.edges[edge_id].setdefault(ModelEdgeInfo.TOT_TENSOR_SIZE, 0)
            graph.edges[edge_id].setdefault(ModelEdgeInfo.TENSOR_NAME_LIST, [])

            tensor_name, tensor_size = (
                input_receiver_info[0],
                input_receiver_info[1],
            )

            graph.edges[edge_id][ModelEdgeInfo.TOT_TENSOR_SIZE] += tensor_size

            if tensor_name not in graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST]:
                graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST].append(tensor_name)

        for output_generator_info in output_generator_info_list:
            tensor_name, tensor_size = (
                output_generator_info[0],
                output_generator_info[1],
            )

            edge_id = (
                NodeId(onnx_node.name),
                NodeId(AbsModelProfiler.OUTPUT_RECEIVER_NAME),
            )

            graph.edges[edge_id].setdefault(ModelEdgeInfo.TOT_TENSOR_SIZE, 0)
            graph.edges[edge_id].setdefault(ModelEdgeInfo.TENSOR_NAME_LIST, [])

            graph.edges[edge_id][ModelEdgeInfo.TOT_TENSOR_SIZE] += tensor_size

            if tensor_name not in graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST]:
                graph.edges[edge_id][ModelEdgeInfo.TENSOR_NAME_LIST].append(tensor_name)

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

        _, tempfile_name = tempfile.mkstemp(suffix=".csv")
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
        return tensor_total_size / (1024 * 1024)

    def __init_size_in_bytes(self, elem_type: onnx.TensorProto.DataType) -> int:
        map_elem = TENSOR_TYPE_MAP.get(elem_type)
        if map_elem is not None:
            ## TODO Check if this is working for integers too
            return map_elem.np_dtype.itemsize
        return 0
