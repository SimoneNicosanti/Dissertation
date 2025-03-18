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

        node_dependency_dict: dict[NodeId, set[ComponentId]] = {}
        node_possible_dict: dict[NodeId, set[ComponentId]] = {}
        component_dependency_dict: dict[ComponentId, set[ComponentId]] = {}

        for node_id in top_order:
            node_dependency_dict[node_id] = set()
            node_possible_dict[node_id] = set()

        for node_id in top_order:
            node_info: dict = solved_model_graph.nodes[node_id]
            server_id = node_info[SolvedNodeInfo.NET_NODE_ID]

            node_dependency_set = node_dependency_dict[node_id]
            node_possible_set = node_possible_dict[node_id]

            exclude_set = set()
            for dep_comp_id in node_dependency_set:
                for poss_comp_id in node_possible_set:
                    if poss_comp_id in component_dependency_dict[dep_comp_id]:
                        exclude_set.add(poss_comp_id)

            difference_set = node_possible_set - exclude_set

            if (
                len(difference_set) == 0
                or node_info[SolvedNodeInfo.GENERATOR]
                or node_info[SolvedNodeInfo.RECEIVER]
            ):
                ## No possible component
                ## Generate new component
                curr_comp_idx = next_comp_dict[server_id]
                node_comp = ComponentId(server_id, curr_comp_idx)
                next_comp_dict[server_id] += 1
            else:
                ## Take one component in the difference set
                node_comp = list(difference_set)[0]

            ## Setting determined node comp
            node_info[SolvedNodeInfo.COMPONENT] = node_comp

            ## All descendants node will depend by this component
            for descendant_id in nx.descendants(solved_model_graph, node_id):
                node_dependency_dict[descendant_id].add(node_comp)

            ## Following nodes having the same server can be in the same component
            if not node_info[SolvedNodeInfo.GENERATOR]:
                for next_node_id in solved_model_graph.successors(node_id):
                    neigh_info: dict = solved_model_graph.nodes[next_node_id]

                    if (
                        node_info[SolvedNodeInfo.NET_NODE_ID]
                        == neigh_info[SolvedNodeInfo.NET_NODE_ID]
                    ):
                        ## Same server --> Setting possible component
                        node_possible_dict[next_node_id].add(node_comp)

                parallel_nodes = (
                    set(solved_model_graph.nodes)
                    - nx.descendants(solved_model_graph, node_id)
                    - nx.ancestors(solved_model_graph, node_id)
                )

                ## Parallel nodes having the same server can be in the same component
                for paral_node_id in parallel_nodes:
                    paral_node_info = solved_model_graph.nodes[paral_node_id]

                    if (
                        node_info[SolvedNodeInfo.NET_NODE_ID]
                        == paral_node_info[SolvedNodeInfo.NET_NODE_ID]
                    ):
                        ## Same server --> Setting possible component
                        node_possible_dict[paral_node_id].add(node_comp)

            ## Expanding component dependency
            ## Making sure that the component does not depend by itself
            component_dependency_dict.setdefault(node_comp, set())
            component_dependency_dict[node_comp] = component_dependency_dict[
                node_comp
            ].union(node_dependency_dict[node_id] - set([node_comp]))

        print("Components found >> ", next_comp_dict)

        is_dag = ConnectedComponentsFinder.cycle_test(solved_model_graph)
        if is_dag:
            print("The Components Graph is DAG")
        else:
            print("The Components Graph is not DAG")

    @staticmethod
    def cycle_test(graph: nx.DiGraph):
        component_graph = nx.DiGraph()

        for node_id in graph.nodes:
            node_component = graph.nodes[node_id][SolvedNodeInfo.COMPONENT]

            component_graph.add_node(node_component)

        for node_id in graph.nodes:
            node_component = graph.nodes[node_id][SolvedNodeInfo.COMPONENT]
            for next_node in graph.successors(node_id):
                next_node_component = graph.nodes[next_node][SolvedNodeInfo.COMPONENT]

                if node_component != next_node_component:
                    component_graph.add_edge(node_component, next_node_component)

        is_dag = nx.is_directed_acyclic_graph(component_graph)
        if not is_dag:
            cycles = nx.find_cycle(component_graph)
            print(cycles)

        return nx.is_directed_acyclic_graph(component_graph)
