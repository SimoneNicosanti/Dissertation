from Graph.AssignmentGraph import AssignmentGraph, AssignmentGraphInfo
from Graph.ConnectedComponents import ConnectedComponentsFinder
from Graph.Graph import EdgeId, Graph, NodeId
from Graph.ModelGraph import ModelGraph
from Optimization.SolvedProblemInfo import SolvedProblemInfo


## TODO Try to make the build faster!!
## Too many embedded cycles
class AssignmentGraphBuilder:
    def __init__(self, graph: ModelGraph, solved_problem_info: SolvedProblemInfo):
        self.graph: ModelGraph = graph
        self.solved_problem_info = solved_problem_info

    def build_assignment_graph(self) -> AssignmentGraph:
        ass_graph_name: str = "{}_assignment_graph".format(self.graph.get_graph_name())
        assignment_graph = AssignmentGraph(ass_graph_name)

        ## Net Node Id >> Assigned Sub Graphs
        assignments_dict: dict[NodeId, list[ModelGraph]] = {}
        for net_node_id in self.solved_problem_info.get_used_network_nodes():
            assignments_dict[net_node_id] = self.__build_sub_graphs_for_network_node(
                net_node_id
            )

        for net_node_id, sub_graphs in assignments_dict.items():

            for sub_grah_idx, sub_graph in enumerate(sub_graphs):
                node_id = assignment_graph.build_node_id(sub_graph.get_graph_name())
                assignment_node_info = AssignmentGraphInfo(
                    net_node_id, sub_graph, sub_grah_idx
                )
                assignment_graph.put_node(node_id, assignment_node_info)

        for block_node_id in assignment_graph.get_nodes_id():

            prev_blocks_ids = self.__find_prev_blocks_ids(
                block_node_id, assignment_graph
            )
            for prev_block_id in prev_blocks_ids:
                edge_id = EdgeId(prev_block_id, block_node_id)
                assignment_graph.put_edge(
                    edge_id,
                    self.graph.get_edge_info(EdgeId(prev_block_id, block_node_id)),
                )

            next_blocks_ids = self.__find_next_blocks_ids(
                block_node_id, assignment_graph
            )
            for next_block_id in next_blocks_ids:
                edge_id = EdgeId(block_node_id, next_block_id)
                assignment_graph.put_edge(
                    edge_id,
                    self.graph.get_edge_info(EdgeId(block_node_id, next_block_id)),
                )

            # for output_edge_id in sub_graph.get_output_edges_id():
            #     ## TODO Find Next Blocks
            #     next_graph_node_id = self.__find_next_graph_node_id(
            #         net_node_id, output_edge_id.second_node_id, assignments_dict
            #     )
            #     if next_graph_node_id is None:
            #         continue
            #     edge_id = EdgeId(node_id, next_graph_node_id)
            #     assignment_graph.put_edge(
            #         edge_id, self.graph.get_edge_info(output_edge_id)
            #     )
        return assignment_graph

    def __find_prev_blocks_ids(
        self, curr_block_id: NodeId, assignment_graph: AssignmentGraph
    ):
        curr_sub_graph: ModelGraph = assignment_graph.get_node_info(
            curr_block_id
        ).get_sub_graph()

        prev_blocks_ids = set()
        for input_edge_id in curr_sub_graph.get_input_edges_id():
            for other_block_id in assignment_graph.get_nodes_id():
                if other_block_id == curr_block_id:
                    continue

                other_sub_graph: ModelGraph = assignment_graph.get_node_info(
                    other_block_id
                ).get_sub_graph()

                if input_edge_id.first_node_id in other_sub_graph.get_nodes_id():
                    prev_blocks_ids.add(other_block_id)

        return prev_blocks_ids

    def __find_next_blocks_ids(
        self, curr_block_id: NodeId, assignment_graph: AssignmentGraph
    ):
        curr_sub_graph: ModelGraph = assignment_graph.get_node_info(
            curr_block_id
        ).get_sub_graph()

        next_blocks_ids = set()
        for output_edge_id in curr_sub_graph.get_output_edges_id():
            for other_block_id in assignment_graph.get_nodes_id():
                if other_block_id == curr_block_id:
                    continue

                other_sub_graph: ModelGraph = assignment_graph.get_node_info(
                    other_block_id
                ).get_sub_graph()

                if output_edge_id.second_node_id in other_sub_graph.get_nodes_id():
                    next_blocks_ids.add(other_block_id)

        return next_blocks_ids

    def __build_sub_graphs_for_network_node(
        self, net_node_id: NodeId
    ) -> list[ModelGraph]:
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

        sub_graphs: list[ModelGraph] = []
        for idx, connected_component in enumerate(connected_components):
            sub_graph: ModelGraph = ModelGraph(
                "{}_server_{}_comp_{}".format(
                    self.graph.get_graph_name(), net_node_id.node_name, idx
                )
            )
            self.__build_sub_graph_from_connected_component(
                sub_graph, connected_component, input_edges, output_edges, self_edges
            )
            sub_graphs.append(sub_graph)

        return sub_graphs

    def __build_sub_graph_from_connected_component(
        self,
        sub_graph: ModelGraph,
        connected_component: list[NodeId],
        input_edges: list[EdgeId],
        output_edges: list[EdgeId],
        self_edges: list[EdgeId],
    ) -> None:

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

                if ModelGraph.is_generator_edge(self_edge_id):
                    sub_graph.put_input_edge(self_edge_id)
                elif ModelGraph.is_receiver_edge(self_edge_id):
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

    def __build_undirect_graph(
        self, assigned_nodes: list[NodeId], self_edges: list[EdgeId]
    ) -> Graph:
        undirect_graph = Graph("Support_Undirect_Graph")
        for mod_node_id in assigned_nodes:
            undirect_graph.put_node(mod_node_id, None)

        for mod_edge_id in self_edges:
            undirect_graph.put_edge(mod_edge_id, None)

            opposite_edge_id = self.graph.build_edge_id(
                mod_edge_id.second_node_id.node_name,
                mod_edge_id.first_node_id.node_name,
            )
            undirect_graph.put_edge(opposite_edge_id, None)

        return undirect_graph
