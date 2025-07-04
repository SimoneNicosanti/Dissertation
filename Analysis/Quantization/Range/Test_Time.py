import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quantize import quantize_static

MODEL_NAME = "yolo11n-seg"

COCO_FILE_PATH = "../../../coco128/preprocessed"


class MyDataReader(CalibrationDataReader):

    def __init__(self, calibration_data: list[np.ndarray]):
        self.idx = 0
        self.tot_data = len(calibration_data)
        self.calibration_data = calibration_data

    def get_next(self):
        if self.idx == self.tot_data:
            return None
        input = self.calibration_data[self.idx]
        self.idx += 1
        return {"images": input}


def build_model_graph():
    with open(f"{MODEL_NAME}.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


def cycle_test(graph: nx.DiGraph, quant_components: set[int]):
    component_graph = nx.DiGraph()

    for node_id in graph.nodes:
        node_component = graph.nodes[node_id]["component"]
        component_graph.add_node(node_component)

        if node_component in quant_components:
            component_graph.nodes[node_component]["is_quant"] = True

    for node_id in graph.nodes:
        node_component = graph.nodes[node_id]["component"]
        for next_node in graph.successors(node_id):
            next_node_component = graph.nodes[next_node]["component"]

            if node_component != next_node_component:
                tensors_names = graph.edges[(node_id, next_node)]["tensor_name_list"]

                if (node_component, next_node_component) in component_graph.edges:
                    component_graph.edges[(node_component, next_node_component)][
                        "tensors_names"
                    ] = component_graph.edges[(node_component, next_node_component)][
                        "tensors_names"
                    ].union(
                        tensors_names
                    )
                else:
                    component_graph.add_edge(
                        node_component,
                        next_node_component,
                        tensors_names=set(tensors_names),
                    )

    is_dag = nx.is_directed_acyclic_graph(component_graph)
    if not is_dag:
        cycles = nx.find_cycle(component_graph)
        print(cycles)

    return nx.is_directed_acyclic_graph(component_graph), component_graph


def divide_in_components(model_graph: nx.DiGraph, quantizable_layers: list[str]):
    top_order: list[str] = list(nx.topological_sort(model_graph))

    node_dependency_dict: dict[str, set[int]] = {}
    node_possible_dict: dict[str, set[int]] = {}
    component_dependency_dict: dict[int, set[int]] = {}

    for node_id in top_order:
        node_dependency_dict[node_id] = set()
        node_possible_dict[node_id] = set()

    next_comp_idx = 0

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
            or node_info.get("generator", False)
            or node_info.get("receiver", False)
            or node_id in quantizable_layers
        ):
            ## No possible component
            ## Generate new component
            node_comp = next_comp_idx
            next_comp_idx += 1

            if node_id in quantizable_layers:
                quant_components.add(node_comp)
        else:
            ## Take one component in the difference set
            node_comp = list(difference_set)[0]

        ## Setting determined node comp
        node_info["component"] = node_comp

        ## All descendants node will depend by this component
        for descendant_id in nx.descendants(model_graph, node_id):
            node_dependency_dict[descendant_id].add(node_comp)

        ## Following nodes having the same server can be in the same component
        if not node_info.get("generator", False) and node_id not in quantizable_layers:
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

    is_dag, comp_graph = cycle_test(model_graph, quant_components)
    if is_dag:
        print("The Components Graph is DAG")
    else:
        print("The Components Graph is not DAG")

    pos = nx.spring_layout(comp_graph, seed=10)  # layout ordinato

    plt.figure(figsize=(9, 9))
    nx.draw(
        comp_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    plt.savefig("comp_graph.png")

    print([comp_graph.edges[edge]["tensors_names"] for edge in comp_graph.edges])

    return comp_graph


def divide_normal(comp_graph: nx.DiGraph):
    top_sort = list(nx.topological_sort(comp_graph))

    for idx, comp in enumerate(top_sort):
        if idx == 0 or idx == len(top_sort) - 1:
            continue

        input_edges = comp_graph.in_edges(comp)
        output_edges = comp_graph.out_edges(comp)

        input_names = set()
        for edge in input_edges:
            input_names = input_names.union(comp_graph.edges[edge]["tensors_names"])

        output_names = set()
        for edge in output_edges:
            output_names = output_names.union(comp_graph.edges[edge]["tensors_names"])

        # print("Input names >> ", input_names)
        # print("Output names >> ", output_names)

        onnx.utils.extract_model(
            "yolo11n-seg.onnx",
            f"yolo11n-seg_comp_{idx}.onnx",
            input_names,
            output_names,
        )


def divide_quantized(comp_graph: nx.DiGraph):

    top_sort = list(nx.topological_sort(comp_graph))

    for idx, comp in enumerate(top_sort):
        if idx == 0 or idx == len(top_sort) - 1:
            continue

        input_edges = comp_graph.in_edges(comp)
        output_edges = comp_graph.out_edges(comp)

        input_names = set()
        for edge in input_edges:
            input_names = input_names.union(comp_graph.edges[edge]["tensors_names"])

        output_names = set()
        for edge in output_edges:
            output_names = output_names.union(comp_graph.edges[edge]["tensors_names"])

        new_output_names = set()
        for output_name in output_names:
            new_output_names.add(output_name + "_DequantizeLinear_Output")

        # print("Input names >> ", input_names)
        # print("Output names >> ", output_names)

        if comp_graph.nodes[comp].get("is_quant", False):

            onnx.utils.extract_model(
                "yolo11n-seg_quant.onnx",
                f"yolo11n-seg_comp_{idx}_quant.onnx",
                input_names,
                new_output_names,
            )


def read_all_images():

    files = os.listdir(COCO_FILE_PATH)

    # Elenca solo i file (non le directory)
    files = [f for f in files if os.path.isfile(os.path.join(COCO_FILE_PATH, f))]
    images = []
    for file_name in files:
        file_path = os.path.join(COCO_FILE_PATH, file_name)
        image = np.load(file_path)["arr_0"]

        images.append(image)

    return images


def main():
    model_graph = build_model_graph()

    sorted_by_flops = sorted(
        model_graph.nodes,
        key=lambda x: model_graph.nodes[x]["flops"],
        reverse=True,
    )
    quantizable_layers = sorted_by_flops[:10]

    for elem in quantizable_layers:
        print(elem, model_graph.nodes[elem]["flops"])

    quantizable_layers.sort()  ## Sorted by name

    comp_graph = divide_in_components(model_graph, quantizable_layers)

    divide_normal(comp_graph)

    # quantize_static(
    #     "yolo11n-seg.onnx",
    #     "yolo11n-seg_quant.onnx",
    #     nodes_to_quantize=quantizable_layers,
    #     extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    #     calibration_data_reader=MyDataReader(read_all_images()),
    # )

    divide_quantized(comp_graph)


if __name__ == "__main__":
    main()
