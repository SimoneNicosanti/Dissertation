from Graph.Graph import Edge, Graph, Node
from Profiler.GraphProfile import GraphProfile


class NetworkGraph(Graph):

    def __init__(self, network_profile: GraphProfile):
        super().__init__()

        self.model_profile: GraphProfile = network_profile
        self.init_graph()

    def init_graph(self):

        for nodeId, nodeInfo in self.model_profile.get_node_profiles().items():
            node = Node(nodeId, nodeInfo)
            self.nodes.append(node)

        for edgeId, edgeInfo in self.model_profile.get_edge_profiles().items():
            edge = Edge(edgeId, edgeInfo)
            self.edges.append(edge)
