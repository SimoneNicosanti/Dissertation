import tempfile

import networkx as nx
import numpy as np
import onnx
import onnxruntime as ort

from CommonIds.NodeId import NodeId
from CommonProfile.ModelInfo import ModelEdgeInfo


class GraphRunner:
    def __init__(
        self, component_graph: nx.DiGraph, model_path: str, quant_model_path: str
    ) -> None:

        self.comp_models: dict[int, onnx.ModelProto] = {}
        self.comp_models_quant: dict[int, onnx.ModelProto] = {}

        self.__extract_sub_models(component_graph, model_path)
        # print("Done Extract 1")
        self.__extract_sub_models(component_graph, quant_model_path, True)
        # print("Done Extract 2")

        self.comp_graph = component_graph

        self.run_input = self.comp_graph.graph["input_names"]

        self.sess = None

        pass

    def __extract_sub_models(
        self,
        comp_graph: nx.DiGraph,
        model_path: str,
        quantizing: bool = False,
    ):
        top_sort = list(nx.topological_sort(comp_graph))

        for idx, comp in enumerate(top_sort):
            if idx == 0 or idx == len(top_sort) - 1:
                ## Input or Output Component
                continue

            input_edges = comp_graph.in_edges(comp)
            output_edges = comp_graph.out_edges(comp)

            input_names = set()
            for edge in input_edges:
                input_names = input_names.union(
                    comp_graph.edges[edge][ModelEdgeInfo.TENSOR_NAME_LIST]
                )

            output_names = set()
            for edge in output_edges:
                output_names = output_names.union(
                    comp_graph.edges[edge][ModelEdgeInfo.TENSOR_NAME_LIST]
                )

            if quantizing:
                new_output_names = set()
                for output_name in output_names:
                    new_output_names.add(output_name + "_DequantizeLinear_Output")
                output_names = new_output_names

            if quantizing and not comp_graph.nodes[comp].get("is_quant_comp", False):
                ## If quantized case and the component is not a quantized one
                ## Then we do not extract the model
                continue

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
                onnx.utils.extract_model(
                    model_path,
                    temp_file.name,
                    input_names,
                    output_names,
                )

                onnx_model = onnx.load_model(temp_file.name)
                onnx_model = onnx.compose.add_prefix(
                    onnx_model, "comp_{}_".format(comp)
                )

                ## Building Inference Session for this component

                # session = ort.InferenceSession(
                #     temp_file.name, providers=["CUDAExecutionProvider"]
                # )

                if quantizing:
                    self.comp_models_quant[comp] = onnx_model
                else:
                    self.comp_models[comp] = onnx_model

    def is_quantized_component(self, comp_id: int, quantization_list: list[NodeId]):

        quantization_mapping = self.comp_graph.graph["quantization_mapping"]

        return (
            self.comp_graph.nodes[comp_id].get("is_quant_comp", False)
            and quantization_mapping[comp_id] in quantization_list
        )

    def set_quantization_configuration(self, quantization_list: list[NodeId]):
        merged_model = self.merge_models(quantization_list)

        self.sess = ort.InferenceSession(
            merged_model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )

    def merge_models(self, quantization_list: list[NodeId]):

        merged_model = None

        for comp_id in nx.topological_sort(self.comp_graph):
            # print("Running >> ", comp_id)
            if self.comp_graph.nodes[comp_id].get("is_generator_comp", False):
                continue
            if self.comp_graph.nodes[comp_id].get("is_receiver_comp", False):
                continue

            curr_comp_model = None
            if self.is_quantized_component(comp_id, quantization_list):
                curr_comp_model = self.comp_models_quant[comp_id]
            else:
                curr_comp_model = self.comp_models[comp_id]

            ## Init of merged model
            if merged_model is None:

                merged_model = curr_comp_model
                continue

            name_mapping = []
            new_output_value_info_proto = []
            for prev_comp_id in self.comp_graph.predecessors(comp_id):

                ## Skip these cases
                if self.comp_graph.nodes[prev_comp_id].get(
                    "is_generator_comp", False
                ) or self.comp_graph.nodes[prev_comp_id].get("is_receiver_comp", False):
                    continue

                input_names = set()
                for tensor_name in self.comp_graph.edges[(prev_comp_id, comp_id)][
                    ModelEdgeInfo.TENSOR_NAME_LIST
                ]:
                    input_name = f"comp_{prev_comp_id}_" + tensor_name
                    if self.is_quantized_component(prev_comp_id, quantization_list):
                        input_name += "_DequantizeLinear_Output"
                    input_names.add(input_name)

                for tensor in merged_model.graph.value_info:
                    if tensor.name in input_names:
                        new_output_value_info_proto.append(tensor)

                for tensor_name in self.comp_graph.edges[(prev_comp_id, comp_id)][
                    ModelEdgeInfo.TENSOR_NAME_LIST
                ]:
                    first_name = f"comp_{prev_comp_id}_" + tensor_name
                    second_name = f"comp_{comp_id}_" + tensor_name

                    if self.is_quantized_component(prev_comp_id, quantization_list):
                        first_name += "_DequantizeLinear_Output"
                    # if self.is_quantized_component(comp_id, quantization_list):
                    #     second_name += "_DequantizeLinear_Output"

                    name_tuple = (first_name, second_name)
                    name_mapping.append(name_tuple)

            merged_model.graph.output.clear()
            merged_model.graph.output.extend(new_output_value_info_proto)

            if self.is_quantized_component(comp_id, quantization_list):
                comp_model = self.comp_models_quant[comp_id]
            else:
                comp_model = self.comp_models[comp_id]

            merged_model = onnx.compose.merge_models(
                merged_model, comp_model, name_mapping
            )

        # print([tensor.name for tensor in merged_model.graph.output])
        onnx.checker.check_model(merged_model, full_check=True)

        return merged_model

    def run(self, input_dict: dict[str, np.ndarray]):
        # print("Running")

        run_output = self.sess.run(None, {"comp_1_images": input_dict["images"]})

        # tensors_dict = {}
        # for key in input_dict.keys():
        #     tensors_dict[key] = ort.OrtValue.ortvalue_from_numpy(
        #         input_dict[key], "cuda"
        #     )

        # for comp_id in nx.topological_sort(self.comp_graph):
        #     # print("Running >> ", comp_id)
        #     if self.comp_graph.nodes[comp_id].get("is_generator_comp", False):
        #         continue
        #     if self.comp_graph.nodes[comp_id].get("is_receiver_comp", False):
        #         continue

        #     curr_session = None
        #     with_quantization = False
        #     if comp_id not in self.comp_models_quant:
        #         curr_session = self.comp_models[comp_id]
        #         # print("Standard Session")
        #     else:
        #         ## This might be a quantized component
        #         ## We have to check if its node has been quantized
        #         quantization_mapping = self.comp_graph.graph["quantization_mapping"]
        #         component_node = quantization_mapping[comp_id]
        #         if component_node not in quantization_list:
        #             curr_session = self.comp_models[comp_id]
        #             # print("Standard Session")
        #         else:
        #             curr_session = self.comp_models_quant[comp_id]
        #             with_quantization = True
        #             # print("Quantized Session")

        #     curr_input_dict = {}
        #     curr_inp_names = set()
        #     for in_edge in self.comp_graph.in_edges(comp_id):
        #         curr_inp_names = curr_inp_names.union(
        #             self.comp_graph.edges[in_edge][ModelEdgeInfo.TENSOR_NAME_LIST]
        #         )
        #     for input_name in curr_inp_names:
        #         curr_input_dict[input_name] = tensors_dict[input_name]

        #     output_array = curr_session.run_with_ort_values(None, curr_input_dict)
        #     # print("Done Run >> ", comp_id)
        #     output_names = [output.name for output in curr_session.get_outputs()]

        #     for idx, output_name in enumerate(output_names):
        #         tensor_name = output_name
        #         if with_quantization:
        #             tensor_name = output_name.replace("_DequantizeLinear_Output", "")
        #         tensors_dict[tensor_name] = output_array[idx]

        # run_output = []
        # for output_name in self.comp_graph.graph["output_names"]:
        #     run_output.append(tensors_dict[output_name].numpy())

        return run_output
