import os

import onnx
from Graph.Graph import NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph
from Graph.SolvedModelGraph import (
    ComponentId,
    SolvedEdgeInfo,
    SolvedModelGraph,
    SolvedNodeInfo,
)
from Partitioner.ModelPartitioner import ModelPartitioner


class OnnxModelPartitioner(ModelPartitioner):

    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.onnx_model: onnx.ModelProto = onnx.load_model(self.model_path)

    def partition_model(self, solved_graph: SolvedModelGraph) -> list[str]:

        for component_id in solved_graph.get_all_components():
            all_comp_nodes: set[NodeId] = solved_graph.get_all_nodes_in_component(
                component_id
            )

            if self.__sub_graph_is_empty(all_comp_nodes):
                continue

            input_names, inp_comp_set = self.__find_input_names(
                all_comp_nodes, solved_graph
            )
            output_names, out_comp_set = self.__find_output_names(
                all_comp_nodes, solved_graph
            )

            model_file_name = os.path.basename(self.model_path)
            comp_file_name = model_file_name.replace(
                ".onnx",
                f"_server_{component_id.net_node_id}_comp_{component_id.component_idx}.onnx",
            )
            output_path = self.model_path.replace(model_file_name, comp_file_name)
            print(component_id)
            print("Inputs")
            for inp_name in input_names:
                print(f"\t{inp_name}")

            print("Outputs")
            for out_name in output_names:
                print(f"\t{out_name}")
            print("Next Components")
            print(f"\t{out_comp_set}")
            print("----------------------")

            onnx.utils.extract_model(
                self.model_path,
                output_path,
                input_names,
                output_names,
            )

        pass

    def __sub_graph_is_empty(self, comp_nodes_set: set[NodeId]) -> bool:
        if len(comp_nodes_set) == 1:
            single_node: NodeId = list(comp_nodes_set)[0]
            if ModelGraph.is_generator_node(single_node):
                return True
            if ModelGraph.is_receiver_node(single_node):
                return True

        return False

    def __find_input_names(
        self, comp_nodes: set[NodeId], solved_graph: SolvedModelGraph
    ) -> list[str]:
        input_names = set()
        comp_sets = set()

        for edge_id in solved_graph.get_edges_id():
            edge_info: SolvedEdgeInfo = solved_graph.get_edge_info(edge_id)
            if edge_id.second_node_id in comp_nodes:

                if (
                    ModelGraph.is_generator_node(edge_id.first_node_id)
                    or edge_id.first_node_id not in comp_nodes
                ):
                    ## Input Node
                    input_names = input_names.union(edge_info.get_tensor_names())

                    if edge_id.first_node_id not in comp_sets:
                        first_node_comp = solved_graph.get_node_info(
                            edge_id.first_node_id
                        ).get_component()
                        comp_sets.add(first_node_comp)

        return input_names, comp_sets

    def __find_output_names(
        self, comp_nodes: set[NodeId], solved_graph: SolvedModelGraph
    ) -> list[str]:
        output_names = set()
        comp_set = set()
        for edge_id in solved_graph.get_edges_id():
            edge_info: SolvedEdgeInfo = solved_graph.get_edge_info(edge_id)

            if edge_id.first_node_id in comp_nodes:
                if (
                    ModelGraph.is_receiver_node(edge_id.second_node_id)
                    or edge_id.second_node_id not in comp_nodes
                ):
                    ## Output Node
                    output_names = output_names.union(edge_info.get_tensor_names())

                    if edge_id.second_node_id not in comp_set:
                        second_node_comp = solved_graph.get_node_info(
                            edge_id.second_node_id
                        ).get_component()
                        comp_set.add(second_node_comp)

        return output_names, comp_set
