class NodeInfo:

    MOD_NODE_FLOPS = 0
    NET_NODE_FLOPS_PER_SEC = 1
    NET_NODE_COMP_ENERGY_PER_SEC = 2
    NET_NODE_TRANS_ENERGY_PER_SEC = 3

    def __init__(self, node_info_dict: dict[int, float]):
        self.node_info_dict = node_info_dict

    def get_info(self, info_key: int) -> float | None:
        return self.node_info_dict.get(info_key, None)


class EdgeInfo:

    MOD_EDGE_DATA_SIZE = 0
    NET_EDGE_BANDWIDTH = 1

    def __init__(self, edge_info_dict: dict[int, float]):
        self.edge_info_dict = edge_info_dict

    def get_info(self, info_key: int) -> float | None:
        return self.edge_info_dict.get(info_key, None)
