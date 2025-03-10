from operator import is_

from Graph.Graph import Graph, NodeId
from Graph.ModelGraph import ModelGraph
from Graph.SolvedModelGraph import (
    ComponentId,
    SolvedModelGraph,
    SolvedNodeInfo,
)


class ConnectedComponentsFinder:

    @staticmethod
    def topological_sort(model_graph: Graph):
        visited = {node_id: False for node_id in model_graph.get_nodes_id()}
        stack = []
        for node_id in model_graph.get_nodes_id():
            if not visited[node_id]:
                ConnectedComponentsFinder.topological_sort_util(
                    model_graph, node_id, visited, stack
                )
        return stack[::-1]  # Inverti la pila per ottenere l'ordine corretto

    @staticmethod
    def topological_sort_util(model_graph: Graph, curr_node_id: NodeId, visited, stack):
        visited[curr_node_id] = True
        for neighbor in model_graph.get_nexts_from_node(curr_node_id):
            if not visited[neighbor]:
                ConnectedComponentsFinder.topological_sort_util(
                    model_graph, neighbor, visited, stack
                )
        stack.append(curr_node_id)  # Push alla fine della visita

    @staticmethod
    def find_connected_components(
        solved_model_graph: SolvedModelGraph,
    ):

        top_order = ConnectedComponentsFinder.topological_sort(solved_model_graph)

        next_comp_idx: dict[NodeId, int] = {}

        for net_node_id in solved_model_graph.get_used_net_nodes():
            next_comp_idx[net_node_id] = 0

        dip_set_dict: dict[NodeId, set[tuple]] = {}
        poss_comp_dict: dict[NodeId, set[tuple]] = {}
        compon_size_count: dict[ComponentId, int] = {}

        for node_id in top_order:
            dip_set_dict[node_id] = set()
            poss_comp_dict[node_id] = set()

        for node_id in top_order:
            node_info: SolvedNodeInfo = solved_model_graph.get_node_info(node_id)
            server_id = node_info.net_node_id
            node_dip_set = dip_set_dict[node_id]
            node_poss_comp_set = poss_comp_dict[node_id]

            if node_info.is_generator():
                ## They will be handled by a different component
                curr_comp_idx = next_comp_idx[server_id]
                node_comp = ComponentId(server_id, curr_comp_idx)
                next_comp_idx[server_id] += 1
                print("Input Component", node_comp)
            elif node_info.is_receiver():
                next_comp_idx[server_id] += 1
                curr_comp_idx = next_comp_idx[server_id]
                node_comp = ComponentId(server_id, curr_comp_idx)
                print("Output Component", node_comp)
            else:
                ## Intermediate node

                if node_dip_set.isdisjoint(node_poss_comp_set):
                    ## No dependency
                    if len(node_poss_comp_set) == 0:
                        ## No possible component
                        ## Generate new component for my server
                        node_comp = ConnectedComponentsFinder.__generate_component_id(
                            server_id, next_comp_idx, node_dip_set
                        )

                    else:
                        ## One or more possible component
                        ## Taking the one with highest component size
                        node_comp = max(
                            node_poss_comp_set, key=lambda comp: compon_size_count[comp]
                        )

                else:
                    ## Dependency
                    ## Generate new component for my server
                    node_comp = ConnectedComponentsFinder.__generate_component_id(
                        server_id, next_comp_idx, node_dip_set
                    )

            ## Setting determined node comp
            node_info.set_component(node_comp)
            compon_size_count.setdefault(node_comp, 0)
            compon_size_count[node_comp] = compon_size_count[node_comp] + 1

            for neigh_id in solved_model_graph.get_nexts_from_node(node_id):
                neigh_info: SolvedNodeInfo = solved_model_graph.get_node_info(neigh_id)

                if node_info.net_node_id == neigh_info.net_node_id and not (
                    node_info.is_generator() or neigh_info.is_receiver()
                ):
                    ## Same server --> Setting possible component
                    poss_comp_dict[neigh_id].add(node_comp)

                else:
                    ## Different server or Generator Node --> Setting dependency
                    dip_set_dict[neigh_id].add(node_comp)

                dip_set_dict[neigh_id] = dip_set_dict[neigh_id].union(
                    dip_set_dict[node_id]
                )

    @staticmethod
    def __generate_component_id(
        server_id: NodeId,
        next_comp_idx_dict: dict[NodeId, int],
        node_dep_set: set[ComponentId],
    ):
        ## If the current component is independent
        ## Use this component
        ## This can handle parallel branches of the model
        curr_comp_idx = next_comp_idx_dict[server_id]
        node_comp = ComponentId(server_id, curr_comp_idx)
        if node_comp not in node_dep_set:
            return node_comp

        ## If there is a dependency with the current component
        ## Generate a new component incrementing the current component index
        next_comp_idx_dict[server_id] += 1
        curr_comp_idx = next_comp_idx_dict[server_id]
        node_comp = ComponentId(server_id, curr_comp_idx)

        return node_comp
