import json

from Optimizer.Graph.Graph import NodeId
from Optimizer.Graph.ModelGraph import ModelGraph
from Optimizer.Graph.SolvedModelGraph import (
    ComponentId,
    SolvedEdgeInfo,
    SolvedModelGraph,
)


class Plan:
    def __init__(self, solved_graph: SolvedModelGraph, deployer_id: str):

        self.plan_dict = {}
        self.solved_graph = solved_graph
        self.deployer_id = deployer_id

        self.__init_plan(solved_graph)

    def dump_plan(self):
        dump_dict = {
            "model_name": self.solved_graph.get_graph_name(),
            "deployer_id": self.deployer_id,
            "plan": self.plan_dict,
        }
        return json.dumps(dump_dict, sort_keys=True)

    def get_all_components(self) -> set[ComponentId]:
        return self.solved_graph.get_all_components()

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

    def __init_plan(self, solved_graph: SolvedModelGraph):
        for component_id in solved_graph.get_all_components():

            key = str(component_id)

            self.plan_dict.setdefault(key, {})

            all_comp_nodes: set[NodeId] = solved_graph.get_all_nodes_in_component(
                component_id
            )

            is_only_input, is_only_output = self.__sub_graph_is_empty(all_comp_nodes)

            self.plan_dict[key]["is_only_input"] = is_only_input
            self.plan_dict[key]["is_only_output"] = is_only_output

            input_names = self.__find_input_names(all_comp_nodes, solved_graph)
            output_connections = self.__find_output_names(
                component_id, all_comp_nodes, solved_graph
            )

            self.plan_dict[key]["input_names"] = list(input_names)
            self.plan_dict[key]["output_connections"] = output_connections

    def __sub_graph_is_empty(self, comp_nodes_set: set[NodeId]) -> bool:
        if len(comp_nodes_set) == 1:
            single_node: NodeId = list(comp_nodes_set)[0]
            if ModelGraph.is_generator_node(single_node):
                return True, False
            if ModelGraph.is_receiver_node(single_node):
                return False, True

        return False, False

    def __find_input_names(
        self, comp_nodes: set[NodeId], solved_graph: SolvedModelGraph
    ) -> list[str]:
        input_names = set()

        for edge_id in solved_graph.get_edges_id():
            edge_info: SolvedEdgeInfo = solved_graph.get_edge_info(edge_id)
            if edge_id.second_node_id in comp_nodes:

                if (
                    ModelGraph.is_generator_node(edge_id.first_node_id)
                    or edge_id.first_node_id not in comp_nodes
                ):
                    ## Input Node
                    input_names = input_names.union(edge_info.get_tensor_names())

            if edge_id.first_node_id in comp_nodes and ModelGraph.is_generator_node(
                edge_id.first_node_id
            ):
                ## If a component contains the input node
                ## Then it has to receive all inputs
                input_names = input_names.union(edge_info.get_tensor_names())

        return input_names

    def __find_output_names(
        self,
        curr_comp_id: ComponentId,
        comp_nodes: set[NodeId],
        solved_graph: SolvedModelGraph,
    ) -> list[str]:
        output_connections: dict[str, set[ComponentId]] = {}

        for edge_id in solved_graph.get_edges_id():
            edge_info: SolvedEdgeInfo = solved_graph.get_edge_info(edge_id)

            if edge_id.first_node_id in comp_nodes and (
                ModelGraph.is_receiver_node(edge_id.second_node_id)
                or edge_id.second_node_id not in comp_nodes
            ):

                second_node_comp = solved_graph.get_node_info(
                    edge_id.second_node_id
                ).get_component()

                ## Output Node
                output_names = edge_info.get_tensor_names()
                for output_name in output_names:
                    output_connections.setdefault(output_name, set())

                    if second_node_comp != curr_comp_id:
                        output_connections[output_name].add(str(second_node_comp))

            if edge_id.second_node_id in comp_nodes and ModelGraph.is_receiver_node(
                edge_id.second_node_id
            ):
                ## If a component contains the output node
                ## Then it has to output all model outputs
                ## But there will be no next component
                output_names = edge_info.get_tensor_names()
                for output_name in output_names:
                    output_connections.setdefault(output_name, set())
                    # output_connections[output_name].append(str(curr_comp_id))

        for output_name in output_connections.keys():
            output_connections[output_name] = list(output_connections[output_name])

        return output_connections
