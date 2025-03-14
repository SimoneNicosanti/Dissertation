import abc
from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class NodeId:
    node_name: str
    node_idx: int

    def __repr__(self) -> str:
        return str(self.node_idx)


@dataclass(frozen=True, repr=False)
class EdgeId:
    first_node_id: NodeId
    second_node_id: NodeId

    def __repr__(self) -> str:
        return "({},{})".format(
            self.first_node_id.node_idx, self.second_node_id.node_idx
        )


class GraphInfo(abc.ABC):
    ## Note
    ## Data Sizes Assumed in MB
    ## Bandwidth Assumed in MB / s
    ## Available Memory Assumed in MB
    @abc.abstractmethod
    def __init__(self):
        super().__init__()


class Graph:

    def __init__(self, graph_name: str):
        self.graph_name = graph_name
        self.node_counter = 0

        self.nodes: dict[NodeId, GraphInfo] = {}
        self.adj_list: dict[NodeId, set[NodeId]] = {}

        self.edges: dict[EdgeId, GraphInfo] = {}

        pass

    def get_nodes_id(self) -> list[NodeId]:
        return list(self.nodes.keys())

    def put_node(self, node_id: NodeId, node_info: GraphInfo):
        if node_id in self.nodes.keys():
            return
        self.nodes[node_id] = node_info

    def build_node_id(self, node_name: str) -> NodeId:
        for node_id in self.get_nodes_id():
            if node_id.node_name == node_name:
                return node_id

        node_id = NodeId(node_name=node_name, node_idx=self.node_counter)
        self.node_counter += 1

        return node_id

    def get_edges_id(self) -> list[EdgeId]:
        return list(self.edges.keys())

    def get_nexts_from_node(self, start_node_id: NodeId) -> list[NodeId]:
        return self.adj_list.get(start_node_id, set())

    def get_prevs_from_node(self, second_node_id: NodeId) -> set[NodeId]:
        prevs_set = set()
        for edge_id in self.get_edges_id():
            if edge_id.second_node_id == second_node_id:
                prevs_set.add(edge_id.first_node_id)

        return prevs_set

    def put_edge(self, edge_id: EdgeId, edge_info: GraphInfo):
        self.adj_list.setdefault(edge_id.first_node_id, set())
        self.adj_list[edge_id.first_node_id].add(edge_id.second_node_id)

        self.edges[edge_id] = edge_info

    def build_edge_id(self, node_name_1: str, node_name_2: str) -> EdgeId:
        node_1 = None
        node_2 = None
        for node_id in self.get_nodes_id():
            if node_id.node_name == node_name_1:
                node_1 = node_id

            if node_id.node_name == node_name_2:
                node_2 = node_id

        if node_1 is None or node_2 is None:
            return None
        edge_id = EdgeId(first_node_id=node_1, second_node_id=node_2)

        return edge_id

    def get_edge_info(self, edge_id: EdgeId) -> GraphInfo | None:
        return self.edges.get(edge_id, None)

    def get_node_info(self, node_id: NodeId) -> GraphInfo | None:
        return self.nodes.get(node_id, None)

    def get_graph_name(self) -> str:
        return self.graph_name
