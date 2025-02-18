from GraphId import EdgeId, NodeId


class NodeInfo:

    def __init__(self, node_id: NodeId, **kwargs):
        self.node_id = node_id
        self.__dict__.update(kwargs)

    def get_info(self, info_name: str):
        return self.__dict__.get(info_name, None)


class EdgeInfo:

    def __init__(self, edge_id: EdgeId, **kwargs):
        self.edge_id = edge_id
        self.__dict__.update(kwargs)

    def get_info(self, info_name: str):
        return self.__dict__.get(info_name, None)


class GraphProfile:

    def __init__(self):
        self.node_profile_map: dict[NodeId, NodeInfo] = {}
        self.edge_profile_map: dict[EdgeId, EdgeInfo] = {}

    def get_node_profiles(self) -> dict[NodeId, NodeInfo]:
        return self.node_profile_map

    def get_edge_profiles(self) -> dict[EdgeId, EdgeInfo]:
        return self.edge_profile_map

    def put_node_profile(self, node_id: NodeId, node_profile: NodeInfo):
        self.node_profile_map[node_id] = node_profile

    def get_node_profile(self, node_id: NodeId) -> NodeInfo:
        return self.node_profile_map.get(node_id, None)

    def get_edge_profile(self, edge_id: EdgeId) -> EdgeInfo:
        return self.edge_profile_map.get(edge_id, None)

    def put_edge_profile(self, edge_id: EdgeId, edge_profile: EdgeInfo):
        self.edge_profile_map[edge_id] = edge_profile
