from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class NodeId:
    node_name: str

    def __repr__(self) -> str:
        return self.node_name


@dataclass(frozen=True, repr=False)
class EdgeId:
    first_node_id: NodeId
    second_node_id: NodeId

    def __repr__(self) -> str:
        return "({})>({})".format(
            self.first_node_id.node_name, self.second_node_id.node_name
        )


class GraphInfo:

    def __init__(self, info_dict: dict[str, float]):
        self.info_dict = info_dict

    def get_info(self, info_key: str) -> float:
        return self.info_dict.get(info_key, 0.0)


class Graph:

    def __init__(self):
        self.nodes: dict[NodeId, GraphInfo] = {}
        self.edges: dict[EdgeId, GraphInfo] = {}

        pass

    def get_nodes_id(self) -> list[NodeId]:
        return list(self.nodes.keys())

    def get_edges_id(self) -> list[EdgeId]:
        return list(self.edges.keys())

    def get_edges_from_start(self, start_node_id: NodeId) -> list[EdgeId]:
        ret_list = []
        for edge_id in self.edges.keys():
            if edge_id.first_node_id == start_node_id:
                ret_list.append(edge_id)

        return ret_list

    def get_edges_from_end(self, end_node_id: NodeId) -> list[EdgeId]:
        ret_list = []
        for edge_id in self.edges.keys():
            if edge_id.second_node_id == end_node_id:
                ret_list.append(edge_id)

        return ret_list

    def put_node(self, node_id: NodeId, node_info: GraphInfo):
        self.nodes[node_id] = node_info

    def put_edge(self, edge_id: EdgeId, edge_info: GraphInfo):
        self.edges[edge_id] = edge_info

    def get_edge_info(self, edge_id: EdgeId) -> GraphInfo | None:
        return self.edges.get(edge_id, None)

    def get_node_info(self, node_id: NodeId) -> GraphInfo | None:
        return self.nodes.get(node_id, None)
