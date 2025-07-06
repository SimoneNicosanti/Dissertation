import copy

import networkx as nx

from CommonIds.NodeId import NodeId
from CommonProfile.ModelInfo import ModelEdgeInfo, ModelNodeInfo


class ComponentDivider:
    def __init__(self) -> None:
        pass

    @staticmethod
    def component_division(model_graph: nx.DiGraph):
        copy_graph = copy.deepcopy(model_graph)
        quant_components, generator_comp, receiver_comp = (
            ComponentDivider.assign_components_to_nodes(copy_graph)
        )

        component_graph: nx.DiGraph = ComponentDivider.build_component_graph(
            copy_graph, quant_components, generator_comp, receiver_comp
        )

        return component_graph

    @staticmethod
    def assign_components_to_nodes(model_graph: nx.DiGraph):
        top_order: list[str] = list(nx.topological_sort(model_graph))

        node_dependency_dict: dict[NodeId, set[int]] = {}
        node_possible_dict: dict[NodeId, set[int]] = {}
        component_dependency_dict: dict[int, set[int]] = {}

        for node_id in top_order:
            node_dependency_dict[node_id] = set()
            node_possible_dict[node_id] = set()

        next_comp_idx = 0
        generator_comp = None
        receiver_comp = None
        quant_components = set()

        for node_id in top_order:
            node_info: dict = model_graph.nodes[node_id]

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
                or node_info.get(ModelNodeInfo.GENERATOR, False)
                or node_info.get(ModelNodeInfo.RECEIVER, False)
                or node_info.get(ModelNodeInfo.QUANTIZABLE, False)
            ):
                ## No possible component
                ## Generate new component
                node_comp = next_comp_idx
                next_comp_idx += 1

                if node_info.get(ModelNodeInfo.QUANTIZABLE, False):
                    quant_components.add(node_comp)

                if node_info.get(ModelNodeInfo.GENERATOR, False):
                    generator_comp = node_comp

                if node_info.get(ModelNodeInfo.RECEIVER, False):
                    receiver_comp = node_comp
            else:
                ## Take one component in the difference set
                node_comp = list(difference_set)[0]

            ## Setting determined node comp
            node_info["component"] = node_comp

            ## All descendants node will depend by this component
            for descendant_id in nx.descendants(model_graph, node_id):
                node_dependency_dict[descendant_id].add(node_comp)

            ## Following nodes having the same server can be in the same component
            if not node_info.get(ModelNodeInfo.GENERATOR, False) and not node_info.get(
                ModelNodeInfo.QUANTIZABLE, False
            ):
                for next_node_id in model_graph.successors(node_id):

                    ## Same server --> Setting possible component
                    node_possible_dict[next_node_id].add(node_comp)

                parallel_nodes = (
                    set(model_graph.nodes)
                    - nx.descendants(model_graph, node_id)
                    - nx.ancestors(model_graph, node_id)
                )

                ## Parallel nodes having the same server can be in the same component
                for paral_node_id in parallel_nodes:

                    ## Same server --> Setting possible component
                    node_possible_dict[paral_node_id].add(node_comp)

            ## Expanding component dependency
            ## Making sure that the component does not depend by itself
            component_dependency_dict.setdefault(node_comp, set())
            component_dependency_dict[node_comp] = component_dependency_dict[
                node_comp
            ].union(node_dependency_dict[node_id] - set([node_comp]))

        print("Components found >> ", next_comp_idx)

        return quant_components, generator_comp, receiver_comp

    @staticmethod
    def build_component_graph(
        model_graph: nx.DiGraph, quant_components, generator_comp, receiver_comp
    ):

        component_graph = nx.DiGraph()

        generator_node = None
        receiver_node = None

        quantization_mapping = {}

        ## Adding components to components graph
        for node_id in model_graph.nodes:
            node_component = model_graph.nodes[node_id]["component"]
            component_graph.add_node(node_component)

            if node_component in quant_components:
                component_graph.nodes[node_component]["is_quant_comp"] = True
                quantization_mapping[node_component] = node_id

            if node_component == generator_comp:
                component_graph.nodes[node_component]["is_generator_comp"] = True

            if node_component == receiver_comp:
                component_graph.nodes[node_component]["is_receiver_comp"] = True

            ## Setting Generator
            if model_graph.nodes[node_id].get(ModelNodeInfo.GENERATOR, False):
                generator_node = node_id

            ## Setting Receiver
            if model_graph.nodes[node_id].get(ModelNodeInfo.RECEIVER, False):
                receiver_node = node_id

        ## Adding edges to components graph
        for node_id in model_graph.nodes:
            node_component = model_graph.nodes[node_id]["component"]
            for next_node in model_graph.successors(node_id):
                next_node_component = model_graph.nodes[next_node]["component"]

                if node_component != next_node_component:
                    tensors_names = model_graph.edges[(node_id, next_node)][
                        ModelEdgeInfo.TENSOR_NAME_LIST
                    ]

                    if (node_component, next_node_component) in component_graph.edges:
                        curr_names: set[str] = component_graph.edges[
                            (node_component, next_node_component)
                        ][ModelEdgeInfo.TENSOR_NAME_LIST]

                        new_names = curr_names.union(tensors_names)

                        component_graph.edges[(node_component, next_node_component)][
                            ModelEdgeInfo.TENSOR_NAME_LIST
                        ] = new_names
                    else:
                        component_graph.add_edge(
                            node_component,
                            next_node_component,
                            tensor_name_list=set(tensors_names),
                        )

        # for edge in component_graph.edges:
        #     print(component_graph.edges[edge][ModelEdgeInfo.TENSOR_NAME_LIST])

        ## Getting input and output names
        input_names = set()
        output_names = set()

        for input_edge in model_graph.out_edges(generator_node):
            input_names = input_names.union(
                model_graph.edges[input_edge][ModelEdgeInfo.TENSOR_NAME_LIST]
            )

        for output_edge in model_graph.in_edges(receiver_node):
            output_names = output_names.union(
                model_graph.edges[output_edge][ModelEdgeInfo.TENSOR_NAME_LIST]
            )

        component_graph.graph["input_names"] = input_names
        component_graph.graph["output_names"] = output_names

        component_graph.graph["quantization_mapping"] = quantization_mapping

        return component_graph
