import abc

from Graph.Graph import Graph, Node


class ModelGraph(Graph, abc.ABC):

    def __init__(self):
        super().__init__()
        self.enter_nodes: list[Node] = []

        pass

    def put_input_node(self, node: Node):
        self.enter_nodes.append(node)

    def get_input_nodes(self):
        return self.enter_nodes
