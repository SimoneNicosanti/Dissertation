import csv
import os
import tempfile

import numpy as np
import onnx
import onnx_tool
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

            if graph.is_in_graph_by_name(onnx_node.name):

                ## profile flops
                node_flops: float = flops_dict[onnx_node.name]

                ## profile weights size
                weights_size: float = self.profile_weights_size_per_node(
                    onnx_node, infered_model
                )

                ## profile out edges size and total output size
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

                node_id = graph.build_node_id(onnx_node.name)
                node_info: ModelNodeInfo = graph.get_node_info(node_id)

                node_info.model_node_flops = node_flops
                node_info.model_node_weights_size = weights_size
                node_info.model_node_total_output_size = total_output_size

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
            ## TODO Manage Weights For Quantized Operators

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

    # ##################

    # def profile_nodes(
    #     self,
    #     graph: ModelGraph,
    #     model: onnx.ModelProto,
    #     auxiliary_edge_dict: dict[str, set[str]],
    # ):

    #     ## Profiling nodes info
    #     flops_dict: dict[str, float] = self.profile_all_flops(model)
    #     weights_size_dict: dict[str, np.ndarray] = self.profile_node_weights_size(model)
    #     output_size_dict: dict[str, np.ndarray] = self.profile_node_outputs_size(model)

    #     ## Getting info only for reachable nodes!! (Excluding Quantize/Dequantize of Weights)
    #     for node_id in graph.get_nodes_id():
    #         node_info: ModelNodeInfo = graph.get_node_info(node_id)
    #         node_info.model_node_flops = flops_dict[node_id.node_name]
    #         node_info.model_node_weights_size = weights_size_dict[node_id.node_name]
    #         node_info.model_node_outputs_size = output_size_dict[node_id.node_name]

    # def profile_node_weights_size(
    #     self, model: onnx.ModelProto, auxiliary_edge_dict: dict[str, set[str]]
    # ) -> dict[str, np.ndarray]:

    #     weights_size_dict: dict[str, float] = {}
    #     for initializer in model.graph.initializer:
    #         weights_size_dict[initializer.name] = self.compute_tensor_size(
    #             [
    #                 onnx.TensorShapeProto.Dimension(dim_value=d)
    #                 for d in initializer.dims
    #             ],
    #             initializer.data_type,
    #         )

    #     node_weights_size_dict: dict[str, float] = {}
    #     for node in model.graph.node:
    #         node_weights_size_dict.setdefault(node.name, 0)
    #         for input_name in node.input:
    #             if input_name in weights_size_dict:
    #                 if node.op_type == "DequantizeLinear":
    #                     ## This is done in order to handle weights quantization --> These are info for the following node
    #                     next_node_name = auxiliary_edge_dict[node.name][0]
    #                     node_weights_size_dict[next_node_name] += weights_size_dict[
    #                         input_name
    #                     ]
    #                     pass
    #                 else:
    #                     ## Normal Case
    #                     node_weights_size_dict[node.name] += weights_size_dict[
    #                         input_name
    #                     ]
    #     return node_weights_size_dict

    # def profile_node_outputs_size(
    #     self, model: onnx.ModelProto
    # ) -> dict[str, np.ndarray]:

    #     infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
    #         model, data_prop=True
    #     )

    #     output_size_dict: dict[str, float] = {}
    #     for value_info in infered_model.graph.value_info:
    #         output_size_dict[value_info.name] = self.compute_tensor_size(
    #             value_info.type.tensor_type.shape.dim,
    #             value_info.type.tensor_type.elem_type,
    #         )

    #     node_output_size_dict: dict[str, float] = {}
    #     for node in infered_model.graph.node:
    #         node_output_size_dict.setdefault(node.name, 0)
    #         for output_name in node.output:
    #             if output_name in output_size_dict:
    #                 node_output_size_dict[node.name] += output_size_dict[output_name]

    #     return node_output_size_dict

    # def profile_edges(self, graph: ModelGraph, model: onnx.ModelProto):

    #     edges_data_info: dict[tuple[str, str], tuple] = self.profile_edges_1()
    #     input_edges_info: dict[str, tuple] = self.profile_input_edges()
    #     output_edges_info: dict[str, tuple] = self.profile_output_edges()

    # def profile_edges_1(self, graph: ModelGraph, model: onnx.ModelProto):
    #     infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
    #         model, data_prop=True
    #     )

    #     tensor_dict: dict[str, onnx.TypeProto.Tensor] = {}
    #     tensor_info: onnx.ValueInfoProto

    #     ## Init from inference
    #     for tensor_info in infered_model.graph.value_info:
    #         tensor_dict[tensor_info.name] = tensor_info.type.tensor_type

    #     edge_data_size_dict: dict[tuple[str, str], float] = {}
    #     edge_data_names_dict: dict[tuple[str, str], float] = {}
    #     for node in infered_model.graph.node:
    #         for other_node in infered_model.graph.node:
    #             for out_name in node.output:
    #                 if out_name in other_node.input:
    #                     out_info = tensor_dict[out_name]

    #     for node in infered_model.graph.node:
    #         for prev_node in infered_model.graph.node:
    #             for inp_name in node.input:
    #                 if inp_name in prev_node.output:
    #                     tensor_info = tensor_dict[inp_name]
    #                     tensor_data_size: float = self.compute_tensor_size(
    #                         tensor_info.shape.dim, tensor_info.elem_type
    #                     )
    #                     edge_id: EdgeId = graph.build_edge_id(prev_node.name, node.name)
    #                     edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
    #                     edge_info.put_tensor_name(inp_name)
    #                     graph.put_edge(edge_id, edge_info)

    #     # --------------------------

    #     input_graph_dict: dict[str, onnx.TypeProto.Tensor] = {
    #         input.name: input.type.tensor_type for input in infered_model.graph.input
    #     }
    #     for node in infered_model.graph.node:
    #         for inp_name in node.input:
    #             if inp_name in input_graph_dict.keys():
    #                 input_tensor = input_graph_dict[inp_name]
    #                 tensor_data_size: float = self.compute_tensor_size(
    #                     input_tensor.shape.dim, input_tensor.elem_type
    #                 )
    #                 edge_id = graph.build_edge_id(
    #                     ModelGraph.INPUT_GENERATOR_NODE_NAME, node.name
    #                 )

    #                 edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
    #                 edge_info.put_tensor_name(inp_name)

    #                 graph.put_edge(edge_id, edge_info)
    #                 graph.put_input_edge(edge_id)

    #     output_graph_dict: dict[str, onnx.TypeProto.Tensor] = {
    #         output.name: output.type.tensor_type
    #         for output in infered_model.graph.output
    #     }
    #     for node in infered_model.graph.node:
    #         for out_name in node.output:
    #             if out_name in output_graph_dict.keys():
    #                 output_tensor = output_graph_dict[out_name]
    #                 tensor_data_size: float = self.compute_tensor_size(
    #                     output_tensor.shape.dim, output_tensor.elem_type
    #                 )
    #                 edge_id = graph.build_edge_id(
    #                     node.name, ModelGraph.OUTPUT_RECEIVER_NODE_NAME
    #                 )

    #                 edge_info = ModelEdgeInfo(model_edge_data_size=tensor_data_size)
    #                 edge_info.put_tensor_name(out_name)
    #                 graph.put_edge(edge_id, edge_info)
    #                 graph.put_output_edge(edge_id)
