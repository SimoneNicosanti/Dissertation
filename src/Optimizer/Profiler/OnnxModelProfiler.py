import csv
import tempfile

import numpy as np
import onnx
import onnx_tool
from GraphId import EdgeId, NodeId
from Profiler.GraphProfile import EdgeInfo, GraphProfile, NodeInfo
from Profiler.ModelProfiler import ModelProfiler


class OnnxModelProfiler(ModelProfiler):

    def profile(
        self, model: onnx.ModelProto, input_shapes: dict[str, tuple]
    ) -> GraphProfile:

        profile: GraphProfile = GraphProfile()
        self.profile_nodes(profile, model, input_shapes)
        self.profile_edges(profile, model, input_shapes)

        return profile

    def profile_nodes(
        self,
        profile: GraphProfile,
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

        with open(tempfile_name, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0 or row[0] == "Total":
                    continue

                name, flops = row[0], row[2]
                node_id: NodeId = NodeId(name)
                node_profile = NodeInfo(node_id=node_id, node_flops=float(flops))

                profile.put_node_profile(node_id, node_profile)

    def profile_edges(
        self,
        profile: GraphProfile,
        model: onnx.ModelProto,
        input_shapes: dict[str, tuple],
    ):

        infered_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
            model, data_prop=True
        )

        tensor_shapes_dict: dict[str, onnx.TypeProto.Tensor] = {}

        value_info_elem: onnx.ValueInfoProto
        for value_info_elem in infered_model.graph.value_info:
            tensor_shapes_dict[value_info_elem.name] = (
                value_info_elem.type.tensor_type.shape
            )

        for node in infered_model.graph.node:
            for prev_node in infered_model.graph.node:
                for inp in node.input:
                    if inp in prev_node.output:

                        tensor_shape: onnx.TensorShapeProto = tensor_shapes_dict[inp]
                        tensor_total_elems = 1

                        for dim in tensor_shape.dim:
                            if dim.HasField("dim_param"):
                                ## Batch Size
                                continue
                            tensor_total_elems *= dim.dim_value

                        edge_id: EdgeId = EdgeId(
                            NodeId(prev_node.name), NodeId(node.name)
                        )
                        edge_profile = EdgeInfo(
                            edge_id=edge_id, data_size=tensor_total_elems * 32
                        )
                        profile.put_edge_profile(edge_id, edge_profile)
