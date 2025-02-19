from Graph.GraphId import EdgeId, NodeId
from Graph.GraphInfo import EdgeInfo, NodeInfo


class Edge:

    def __init__(self, edge_id: EdgeId, edge_info: EdgeInfo):
        self.edge_id = edge_id
        self.edge_info = edge_info

    def get_edge_info(self) -> EdgeInfo:
        return self.edge_info

    def get_edge_id(self) -> EdgeId:
        return self.edge_id


class Node:

    def __init__(self, node_id: NodeId, node_info: NodeInfo):
        self.node_id = node_id
        self.node_info: NodeInfo = node_info
        pass

    def get_node_info(self) -> NodeInfo:
        return self.node_info

    def get_node_id(self) -> NodeId:
        return self.node_id


class Graph:

    def __init__(self):
        self.nodes: dict[NodeId, Node] = {}
        self.edges: dict[NodeId, Node] = {}

        pass

    def get_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def get_edges(self) -> list[Edge]:
        return list(self.edges.values())

    def put_node(self, node_id: NodeId, node: Node):
        self.nodes[node_id] = node

    def put_edge(self, edge_id: EdgeId, edge: Edge):
        self.edges[edge_id] = edge

    def get_edge(self, edge_id: EdgeId) -> Edge | None:
        return self.edges.get(edge_id, None)
