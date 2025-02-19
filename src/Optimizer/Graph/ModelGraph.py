from Graph.Graph import Edge, Graph, Node
from Graph.GraphId import EdgeId, NodeId


class ModelGraph(Graph):

    def __init__(self):
        super().__init__()
        self.enter_nodes: list[NodeId] = []

        pass

    def set_enter_nodes(self, enter_nodes : list[NodeId]) :
        for node_id in enter_nodes :
            if node_id in self.nodes.keys() :
                self.enter_nodes.append(node_id)
        
            
