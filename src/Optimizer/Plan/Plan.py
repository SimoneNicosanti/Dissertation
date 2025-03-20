import json

import networkx as nx

from CommonProfile.NodeId import NodeId
from Optimizer.Graph.SolvedModelGraph import SolvedNodeInfo, SolvedEdgeInfo

from Optimizer.Graph.SolvedModelGraph import (
    ComponentId,
)


class Plan:
    def __init__(self, solved_graph: nx.DiGraph, deployer_id: str):

        self.plan_dict = {}
        self.solved_graph = solved_graph
        self.deployer_id = deployer_id

        self.__init_plan()

    def dump_plan(self):
        dump_dict = {
            "model_name": self.solved_graph.graph["name"],
            "deployer_id": self.deployer_id,
            "plan": self.plan_dict,
        }
        return json.dumps(dump_dict, sort_keys=True)

    def get_all_components(self) -> set[ComponentId]:
        components_set = set()
        for _, data in self.solved_graph.nodes(data=True):
            components_set.add(data[SolvedNodeInfo.COMPONENT])

        return components_set

    def is_component_only_input(self, component_id: ComponentId) -> bool:
        key = str(component_id)
        return self.plan_dict[key]["is_only_input"]

    def is_component_only_output(self, component_id: ComponentId) -> bool:
        key = str(component_id)
        return self.plan_dict[key]["is_only_output"]

    def get_input_names_per_component(self, component_id: ComponentId) -> list[str]:
        key = str(component_id)
        return self.plan_dict[key]["input_names"]

    def get_output_names_per_component(self, component_id: ComponentId) -> list[str]:
        key = str(component_id)
        return self.plan_dict[key]["output_connections"].keys()

    def __get_all_nodes_in_component(self, component_id: ComponentId) -> set[NodeId]:
        nodes_set: set[NodeId] = set()
        for node, data in self.solved_graph.nodes(data=True):
            if data[SolvedNodeInfo.COMPONENT] == component_id:
                nodes_set.add(node)
        return nodes_set

    def __init_plan(self):
        for component_id in self.get_all_components():

            key = str(component_id)

            self.plan_dict.setdefault(key, {})

            all_comp_nodes: set[NodeId] = self.__get_all_nodes_in_component(
                component_id
            )

            is_only_input, is_only_output = self.__sub_graph_is_empty(all_comp_nodes)

            self.plan_dict[key]["is_only_input"] = is_only_input
            self.plan_dict[key]["is_only_output"] = is_only_output

            input_names = self.__find_input_names(all_comp_nodes, component_id)
            output_connections = self.__find_output_names(all_comp_nodes, component_id)

            self.plan_dict[key]["input_names"] = list(input_names)
            self.plan_dict[key]["output_connections"] = output_connections

    def __sub_graph_is_empty(self, comp_nodes_set: set[NodeId]) -> bool:
        if len(comp_nodes_set) == 1:
            single_node_id: NodeId = list(comp_nodes_set)[0]
            single_node_data = self.solved_graph.nodes[single_node_id]
            if single_node_data[SolvedNodeInfo.GENERATOR]:
                return True, False
            if single_node_data[SolvedNodeInfo.RECEIVER]:
                return False, True

        return False, False

    def __find_input_names(
        self, comp_nodes: set[NodeId], component_id: ComponentId
    ) -> list[str]:
        input_names = set()

        for node_id in comp_nodes:
            prev_nodes = self.solved_graph.predecessors(node_id)

            for prev_node_id in prev_nodes:
                prev_comp_id: ComponentId = self.solved_graph.nodes[prev_node_id][
                    SolvedNodeInfo.COMPONENT
                ]

                ## It has to receive all tensors from other components
                if prev_comp_id != component_id:
                    edge_id = (prev_node_id, node_id)
                    tensor_names_list = self.solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    input_names = input_names.union(tensor_names_list)

            ## If the component contains the generator node
            ## It has to receive all the inputs
            if self.solved_graph.nodes[node_id][SolvedNodeInfo.GENERATOR]:
                out_edges = self.solved_graph.out_edges(node_id)

                for edge_id in out_edges:
                    tensor_names_list = self.solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    input_names = input_names.union(tensor_names_list)

        return list(input_names)

    def __find_output_names(
        self,
        comp_nodes: set[NodeId],
        component_id: ComponentId,
    ) -> list[str]:
        ## TensorName --> List of receiver components
        output_connections: dict[str, set[ComponentId]] = {}

        for node_id in comp_nodes:
            next_nodes = self.solved_graph.successors(node_id)
            for next_node_id in next_nodes:
                next_comp_id: ComponentId = self.solved_graph.nodes[next_node_id][
                    SolvedNodeInfo.COMPONENT
                ]

                if next_comp_id != component_id:
                    edge_id = (node_id, next_node_id)
                    tensor_names_list = self.solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    for tensor_name in tensor_names_list:
                        output_connections.setdefault(tensor_name, set())
                        output_connections[tensor_name].add(str(next_comp_id))

            ## If the component contains the receiver node
            ## It has to output all the outputs
            ## But there will be no next component
            if self.solved_graph.nodes[node_id][SolvedNodeInfo.RECEIVER]:
                in_edges = self.solved_graph.in_edges(node_id)

                for edge_id in in_edges:
                    tensor_names_list = self.solved_graph.edges[edge_id][
                        SolvedEdgeInfo.TENSOR_NAME_LIST
                    ]

                    for tensor_name in tensor_names_list:
                        output_connections.setdefault(tensor_name, set())
                        # output_connections[tensor_name].add(str(component_id))

        for output_name in output_connections.keys():
            output_connections[output_name] = list(output_connections[output_name])

        return output_connections
