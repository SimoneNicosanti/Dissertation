from Graph.Graph import Graph, GraphInfo


class NetworkGraph(Graph):

    def __init__(self, graph_name: str):
        super().__init__(graph_name)


class NetworkEdgeInfo(GraphInfo):

    def __init__(self, net_edge_bandwidth: float):
        super().__init__()
        self.net_edge_bandwidth = net_edge_bandwidth

    def get_edge_bandwidth(self):
        return self.net_edge_bandwidth


class NetworkNodeInfo(GraphInfo):

    def __init__(
        self,
        net_node_flops_per_sec: float,
        net_node_comp_energy_per_sec: float,
        net_node_trans_energy_per_sec: float,
        net_node_available_memory: float,
    ):
        super().__init__()
        self.net_node_flops_per_sec = net_node_flops_per_sec
        self.net_node_comp_energy_per_sec = net_node_comp_energy_per_sec
        self.net_node_trans_energy_per_sec = net_node_trans_energy_per_sec
        self.net_node_available_memory = net_node_available_memory

    def get_flops_per_sec(self):
        return self.net_node_flops_per_sec

    def get_comp_energy_per_sec(self):
        return self.net_node_comp_energy_per_sec

    def get_trans_energy_per_sec(self):
        return self.net_node_trans_energy_per_sec

    def get_available_memory(self):
        return self.net_node_available_memory
