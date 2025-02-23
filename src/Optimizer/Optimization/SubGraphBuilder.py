from Graph.ConnectedComponents import ConnectedComponentsFinder
from Graph.Graph import EdgeId, Graph, NodeId
from Graph.ModelGraph import ModelGraph
from Optimization.SolvedProblemInfo import SolvedProblemInfo


class SubGraphBuilder:

    def __init__(self, graph: Graph, solved_problem_info: SolvedProblemInfo) -> None:
        self.graph = graph
        self.solved_problem_info = solved_problem_info
        pass

    def build_sub_graphs(self) -> dict[NodeId, list[ModelGraph]]:
        sub_graphs_by_node: dict[NodeId, list[ModelGraph]] = {}
        for net_node_id in self.solved_problem_info.get_used_network_nodes():
            sub_graphs_by_node[net_node_id] = self.__build_sub_graphs_for_network_node(
                net_node_id
            )

            print(sub_graphs_by_node[net_node_id])

        return sub_graphs_by_node

    def __build_sub_graphs_for_network_node(self, net_node_id: NodeId) -> list[Graph]:
        input_edges: list[EdgeId] = (
            self.solved_problem_info.get_input_edges_for_network_node(net_node_id)
        )
        output_edges: list[EdgeId] = (
            self.solved_problem_info.get_output_edges_for_network_node(net_node_id)
        )
        self_edges: list[EdgeId] = (
            self.solved_problem_info.get_self_edges_for_network_node(net_node_id)
        )

        assigned_nodes: list[NodeId] = (
            self.solved_problem_info.get_assigned_nodes_for_network_node(net_node_id)
        )

        undirect_graph = self.__build_undirect_graph(assigned_nodes, self_edges)
        connected_components: list[list[NodeId]] = (
            ConnectedComponentsFinder.find_connected_components(undirect_graph)
        )
        print(len(connected_components))

        sub_graphs: list[ModelGraph] = []
        for connected_component in connected_components:
            sub_graph: ModelGraph = self.__build_sub_graph_from_connected_component(
                connected_component, input_edges, output_edges, self_edges
            )
            sub_graphs.append(sub_graph)

        return sub_graphs

    def __build_sub_graph_from_connected_component(
        self,
        connected_component: list[NodeId],
        input_edges: list[EdgeId],
        output_edges: list[EdgeId],
        self_edges: list[EdgeId],
    ) -> ModelGraph:
        sub_graph = ModelGraph()

        for node_id in connected_component:
            node_info = self.graph.get_node_info(node_id)
            sub_graph.put_node(node_id, node_info)

        for self_edge_id in self_edges:
            if (
                self_edge_id.first_node_id in connected_component
                and self_edge_id.second_node_id in connected_component
            ):
                edge_info = self.graph.get_edge_info(self_edge_id)
                sub_graph.put_edge(self_edge_id, edge_info)

                if self_edge_id.first_node_id == NodeId(
                    ModelGraph.INPUT_GENERATOR_NODE_NAME
                ):
                    sub_graph.put_input_edge(self_edge_id)
                elif self_edge_id.second_node_id == NodeId(
                    ModelGraph.OUTPUT_RECEIVER_NODE_NAME
                ):
                    sub_graph.put_output_edge(self_edge_id)

        for input_edge_id in input_edges:
            if input_edge_id.second_node_id in connected_component:
                sub_graph.put_edge(
                    input_edge_id, self.graph.get_edge_info(input_edge_id)
                )
                sub_graph.put_input_edge(input_edge_id)

        for output_edge_id in output_edges:
            if output_edge_id.first_node_id in connected_component:
                sub_graph.put_edge(
                    output_edge_id, self.graph.get_edge_info(output_edge_id)
                )
                sub_graph.put_output_edge(output_edge_id)

        return sub_graph

    def __build_undirect_graph(
        self, assigned_nodes: list[NodeId], self_edges: list[EdgeId]
    ) -> Graph:
        undirect_graph = Graph()
        for mod_node_id in assigned_nodes:
            undirect_graph.put_node(mod_node_id, None)

        for mod_edge_id in self_edges:
            undirect_graph.put_edge(mod_edge_id, None)

            opposite_edge_id = EdgeId(
                mod_edge_id.second_node_id, mod_edge_id.first_node_id
            )
            undirect_graph.put_edge(opposite_edge_id, None)

        return undirect_graph
