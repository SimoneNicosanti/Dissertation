import onnx
from Graph.Graph import Edge, Node
from Graph.Model.ModelGraph import ModelGraph
from GraphId import EdgeId, NodeId
from Profiler.GraphProfile import GraphProfile


class OnnxModelGraph(ModelGraph):

    def __init__(self, onnx_model: onnx.ModelProto, model_profile: GraphProfile):
        super().__init__(model_profile)
        self.model: onnx.ModelProto = onnx_model

        self.init_graph()

    def init_graph(self) -> None:
        self.init_nodes()
        self.init_edges()

    def init_nodes(self) -> None:
        for node in self.model.graph.node:
            node_id: NodeId = NodeId(node.name)
            node_profile = self.model_profile.get_node_profile(node_id)

            model_node = Node(node_id, node_profile)

            self.model_nodes.append(model_node)

    def init_edges(self) -> None:
        node: onnx.NodeProto
        prev_node: onnx.NodeProto

        ## TODO >> Refactor this try to remove the three for loops
        for node in self.model.graph.node:
            for prev_node in self.model.graph.node:
                for input in node.input:
                    if input in prev_node.output:
                        edge_id: EdgeId = EdgeId(
                            NodeId(prev_node.name), NodeId(node.name)
                        )
                        edge_profile = self.model_profile.get_edge_profile(edge_id)
                        model_edge = Edge(edge_id, edge_profile)
                        self.model_edges.append(model_edge)
