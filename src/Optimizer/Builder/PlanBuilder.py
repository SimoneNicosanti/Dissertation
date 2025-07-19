import networkx as nx

from CommonIds.ComponentId import ComponentId
from CommonIds.NodeId import NodeId
from CommonPlan.Plan import Plan
from CommonPlan.SolvedModelGraph import SolvedEdgeInfo, SolvedGraphInfo, SolvedNodeInfo


class PlanBuilder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def build_plan(solved_graph: nx.DiGraph) -> Plan:
        model_name = solved_graph.graph["name"]
        plan_dict: dict[ComponentId, dict] = {}

        quantized_list = [
            node
            for node, data in solved_graph.nodes(data=True)
            if data.get(SolvedNodeInfo.QUANTIZED, False)
        ]
        print(quantized_list)

        ## Init Component Keys
        for _, data in solved_graph.nodes(data=True):
            plan_dict.setdefault(data[SolvedNodeInfo.COMPONENT], {})

        ## Init Plan Dict
        for component_id in plan_dict.keys():

            all_comp_nodes: set[NodeId] = PlanBuilder.__get_all_nodes_in_component(
                solved_graph, component_id
            )

            is_only_input, is_only_output = PlanBuilder.__sub_graph_is_empty(
                solved_graph, all_comp_nodes
            )

            plan_dict[component_id]["is_only_input"] = is_only_input
            plan_dict[component_id]["is_only_output"] = is_only_output

            ## TODO Modify to support quantization cuts
            input_names = PlanBuilder.__find_input_names(
                solved_graph, all_comp_nodes, component_id
            )
            output_connections = PlanBuilder.__find_output_names(
                solved_graph, all_comp_nodes, component_id
            )

            plan_dict[component_id]["input_names"] = list(input_names)
            plan_dict[component_id]["output_connections"] = output_connections

        return Plan(
            model_name,
            plan_dict,
            quantized_list,
            solved_graph.graph[SolvedGraphInfo.LATENCY_VALUE],
            solved_graph.graph[SolvedGraphInfo.ENERGY_VALUE],
            solved_graph.graph[SolvedGraphInfo.DEVICE_ENERGY_VALUE],
            solved_graph.graph[SolvedGraphInfo.LATENCY_COST],
            solved_graph.graph[SolvedGraphInfo.ENERGY_COST],
        )

    @staticmethod
    def __get_all_nodes_in_component(
        solved_graph: nx.DiGraph, component_id: ComponentId
    ) -> set[NodeId]:
        nodes_set: set[NodeId] = set()
        for node, data in solved_graph.nodes(data=True):
            if data[SolvedNodeInfo.COMPONENT] == component_id:
                nodes_set.add(node)
        return nodes_set

    @staticmethod
    def __sub_graph_is_empty(
        solved_graph: nx.DiGraph, comp_nodes_set: set[NodeId]
    ) -> bool:
        if len(comp_nodes_set) == 1:
            single_node_id: NodeId = list(comp_nodes_set)[0]
            single_node_data = solved_graph.nodes[single_node_id]
            if single_node_data[SolvedNodeInfo.GENERATOR]:
                return True, False
            if single_node_data[SolvedNodeInfo.RECEIVER]:
                return False, True

        return False, False

    @staticmethod
    def __find_input_names(
        solved_graph: nx.DiGraph,
        comp_nodes: set[NodeId],
        component_id: ComponentId,
    ) -> list[str]:
        input_names = set()

        for node_id in comp_nodes:
            prev_nodes = solved_graph.predecessors(node_id)

            for prev_node_id in prev_nodes:
                prev_comp_id: ComponentId = solved_graph.nodes[prev_node_id][
                    SolvedNodeInfo.COMPONENT
                ]

                ## It has to receive all tensors from other components
                if prev_comp_id != component_id:
                    edge_id = (prev_node_id, node_id)
                    tensor_names_list = solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    input_names = input_names.union(tensor_names_list)

            ## If the component contains the generator node
            ## It has to receive all the inputs
            if solved_graph.nodes[node_id][SolvedNodeInfo.GENERATOR]:
                out_edges = solved_graph.out_edges(node_id)

                for edge_id in out_edges:
                    tensor_names_list = solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    input_names = input_names.union(tensor_names_list)

        return list(input_names)

    @staticmethod
    def __find_output_names(
        solved_graph: nx.DiGraph,
        comp_nodes: set[NodeId],
        component_id: ComponentId,
    ) -> list[str]:
        ## TensorName --> List of receiver components
        output_connections: dict[str, set[ComponentId]] = {}

        for node_id in comp_nodes:
            next_nodes = solved_graph.successors(node_id)
            for next_node_id in next_nodes:
                next_comp_id: ComponentId = solved_graph.nodes[next_node_id][
                    SolvedNodeInfo.COMPONENT
                ]

                if next_comp_id != component_id:
                    edge_id = (node_id, next_node_id)
                    tensor_names_list = solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    for tensor_name in tensor_names_list:
                        output_connections.setdefault(tensor_name, set())
                        output_connections[tensor_name].add(next_comp_id)

            ## If the component contains the receiver node
            ## It has to output all the outputs
            ## But there will be no next component
            if solved_graph.nodes[node_id][SolvedNodeInfo.RECEIVER]:
                in_edges = solved_graph.in_edges(node_id)

                for edge_id in in_edges:
                    tensor_names_list = solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    for tensor_name in tensor_names_list:
                        output_connections.setdefault(tensor_name, set())
                        # output_connections[tensor_name].add(str(component_id))

        for output_name in output_connections.keys():
            output_connections[output_name] = list(output_connections[output_name])

        return output_connections
