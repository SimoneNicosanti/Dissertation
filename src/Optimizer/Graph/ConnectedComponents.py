import networkx as nx

from Optimizer.Graph.Graph import NodeId, SolvedNodeInfo
from Optimizer.Graph.SolvedModelGraph import (
    ComponentId,
)


class ConnectedComponentsFinder:

    @staticmethod
    def find_connected_components(
        solved_model_graph: nx.DiGraph,
    ):

        top_order: list[NodeId] = list(nx.topological_sort(solved_model_graph))

        next_comp_dict: dict[NodeId, int] = {}

        used_net_nodes = set(
            [
                node_info[SolvedNodeInfo.NET_NODE_ID]
                for _, node_info in solved_model_graph.nodes(data=True)
            ]
        )
        for net_node_id in used_net_nodes:
            next_comp_dict[net_node_id] = 0

        dependency_dict: dict[NodeId, set[ComponentId]] = {}
        possible_dict: dict[NodeId, set[ComponentId]] = {}
        component_size_dict: dict[ComponentId, int] = {}

        for node_id in top_order:
            dependency_dict[node_id] = set()
            possible_dict[node_id] = set()

        for node_id in top_order:
            node_info: dict = solved_model_graph.nodes[node_id]
            server_id = node_info[SolvedNodeInfo.NET_NODE_ID]
            node_dipendency_set = dependency_dict[node_id]
            node_possible_set = possible_dict[node_id]

            if node_info[SolvedNodeInfo.GENERATOR]:
                ## They will be handled by a different component
                curr_comp_idx = next_comp_dict[server_id]
                node_comp = ComponentId(server_id, curr_comp_idx)
                next_comp_dict[server_id] += 1

            elif node_info[SolvedNodeInfo.RECEIVER]:
                next_comp_dict[server_id] += 1
                curr_comp_idx = next_comp_dict[server_id]
                node_comp = ComponentId(server_id, curr_comp_idx)

            else:
                ## Intermediate node
                node_diff_set = node_possible_set.difference(node_dipendency_set)

                if len(node_diff_set) == 0:
                    ## No possible component
                    ## Generate new component
                    node_comp = ConnectedComponentsFinder.__generate_component_id(
                        server_id, next_comp_dict, node_dipendency_set
                    )
                else:
                    ## Take one component among the possible
                    ## Taking the component with highest component size
                    node_comp = max(
                        node_diff_set,
                        key=lambda comp: component_size_dict[comp],
                    )

            ## Setting determined node comp
            node_info[SolvedNodeInfo.COMPONENT] = node_comp
            component_size_dict.setdefault(node_comp, 0)
            component_size_dict[node_comp] = component_size_dict[node_comp] + 1

            for neigh_id in solved_model_graph.successors(node_id):
                neigh_info: dict = solved_model_graph.nodes[neigh_id]

                if node_info[SolvedNodeInfo.NET_NODE_ID] == neigh_info[
                    SolvedNodeInfo.NET_NODE_ID
                ] and not (
                    node_info[SolvedNodeInfo.GENERATOR]
                    or neigh_info[SolvedNodeInfo.RECEIVER]
                ):
                    ## Same server --> Setting possible component
                    possible_dict[neigh_id].add(node_comp)

                else:
                    ## Different server or Generator Node --> Setting dependency
                    dependency_dict[neigh_id].add(node_comp)

                dependency_dict[neigh_id] = dependency_dict[neigh_id].union(
                    dependency_dict[node_id]
                )
        print("Components found >> ", next_comp_dict)

        is_dag = ConnectedComponentsFinder.cycle_test(solved_model_graph)
        if is_dag:
            print("The Components Graph is DAG")
        else:
            print("The Components Graph is not DAG")

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

    @staticmethod
    def cycle_test(graph: nx.DiGraph):
        component_graph = nx.DiGraph()

        for node_id in graph.nodes:
            node_component = graph.nodes[node_id][SolvedNodeInfo.COMPONENT]

            component_graph.add_node(node_component)
            component_graph.nodes[node_component].setdefault("nodes", [])
            component_graph.nodes[node_component]["nodes"].append(node_id)

        for node_id in graph.nodes:
            for next_node in graph.successors(node_id):
                next_node_component = graph.nodes[next_node][SolvedNodeInfo.COMPONENT]

                if node_component != next_node_component:
                    component_graph.add_edge(node_component, next_node_component)

        return nx.is_directed_acyclic_graph(component_graph)
