import abc

from Graph.Graph import EdgeId, Graph, NodeId


class ModelGraph(Graph, abc.ABC):

    def __init__(self):
        super().__init__()
        self.input_nodes: list[NodeId] = []
        self.output_edges: list[EdgeId] = []

        self.input_edges: list[NodeId] = []
        self.output_edges: list[EdgeId] = []

        pass

    def put_input_node(self, node: NodeId):
        self.input_nodes.append(node)

    def get_input_nodes(self):
        return self.input_nodes

    def put_output_node(self, node: NodeId):
        self.output_nodes.append(node)

    def get_output_nodes(self):
        return self.output_nodes

    def put_input_edge(self, edge: EdgeId):
        self.input_edges.append(edge)

    def get_input_edges(self):
        return self.input_edges

    def put_output_edge(self, edge: EdgeId):
        self.output_edges.append(edge)

    def get_output_edges(self):
        return self.output_edges
