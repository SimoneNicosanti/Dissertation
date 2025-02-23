from Graph.Graph import Graph, NodeId


class ConnectedComponentsFinder:

    @staticmethod
    def find_connected_components(undirect_graph: Graph) -> list[list[NodeId]]:
        connected_components: list[list[NodeId]] = []
        is_visited = {node_id: False for node_id in undirect_graph.get_nodes_id()}

        for node_id in undirect_graph.get_nodes_id():
            if not is_visited[node_id]:
                component = []
                ConnectedComponentsFinder.__find_connected_component(
                    node_id, is_visited, component, undirect_graph
                )
                connected_components.append(component)

        return connected_components

    # Finds a connected component
    # starting from source using DFS
    @staticmethod
    def __find_connected_component(
        src_node_id: NodeId,
        is_visited: dict[NodeId, bool],
        component: list[NodeId],
        undirected_graph: Graph,
    ):
        is_visited[src_node_id] = True
        component.append(src_node_id)

        for edge_id in undirected_graph.get_edges_from_start(src_node_id):
            v = edge_id.second_node_id
            if not is_visited[v]:
                ConnectedComponentsFinder.__find_connected_component(
                    v, is_visited, component, undirected_graph
                )
