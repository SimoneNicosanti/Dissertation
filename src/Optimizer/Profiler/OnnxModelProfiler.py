import csv
import tempfile

import numpy as np
import onnx
import onnx_tool
from Graph.Graph import Edge, Node
from Graph.GraphId import EdgeId, NodeId
from Graph.GraphInfo import EdgeInfo, NodeInfo
from Graph.ModelGraph import ModelGraph
from Profiler.ModelProfiler import ModelProfiler


class OnnxModelProfiler(ModelProfiler):

    def profile_model(
        self, model: onnx.ModelProto, input_shapes: dict[str, tuple]
    ) -> ModelGraph:

        graph: ModelGraph = ModelGraph()
        self.init_nodes(graph, model, input_shapes)
        self.init_edges(graph, model, input_shapes)

        self.init_enter_nodes(graph, model)

        return graph

    def init_enter_nodes(self, graph: ModelGraph, model: onnx.ModelProto):
        graph.enter_nodes = [NodeId(inp.name) for inp in model.graph.input]

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

        with open(tempfile_name, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0 or row[0] == "Total":
                    continue

                name, flops = row[0], row[2]
                node_id: NodeId = NodeId(name)
                node_info = NodeInfo({NodeInfo.MOD_NODE_FLOPS: float(flops)})
                node: Node = Node(node_id, node_info)

                graph.put_node(node_id, node)

    def init_edges(
        self,
        graph: ModelGraph,
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
                        tensor_total_elems = 1  ## TODO >> Init with data size

                        for dim in tensor_shape.dim:
                            if dim.HasField("dim_param"):
                                ## Batch Size
                                continue
                            tensor_total_elems *= dim.dim_value

                        edge_id: EdgeId = EdgeId(
                            NodeId(prev_node.name), NodeId(node.name)
                        )
                        edge_info = EdgeInfo(
                            {EdgeInfo.MOD_EDGE_DATA_SIZE: tensor_total_elems}
                        )
                        edge = Edge(edge_id, edge_info)
                        graph.put_edge(edge_id, edge)
