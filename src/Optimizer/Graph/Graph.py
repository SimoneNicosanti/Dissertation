from dataclasses import dataclass

from Graph.GraphInfo import EdgeInfo, NodeInfo


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


class Graph:

    def __init__(self):
        self.nodes: dict[NodeId, NodeInfo] = {}
        self.edges: dict[EdgeId, EdgeInfo] = {}

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

    def put_node(self, node_id: NodeId, node_info: NodeInfo):
        self.nodes[node_id] = node_info

    def put_edge(self, edge_id: EdgeId, edge_info: EdgeInfo):
        self.edges[edge_id] = edge_info

    def get_edge_info(self, edge_id: EdgeId) -> EdgeInfo | None:
        return self.edges.get(edge_id, None)

    def get_node_info(self, node_id: NodeId) -> NodeInfo | None:
        return self.nodes.get(node_id, None)
