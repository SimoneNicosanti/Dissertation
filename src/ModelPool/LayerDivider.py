import tempfile

import onnx


class LayerDivider:

    def __init__(self, onnx_model_path: str, quant_onnx_model_path=None) -> None:

        self.quantized = quant_onnx_model_path is not None
        if self.quantized:

            quant_onnx_model = onnx.load_model(quant_onnx_model_path)
            self.quantized_tensors = self.__init_tensors(quant_onnx_model)

            self.model_path = quant_onnx_model_path
        else:
            self.model_path = onnx_model_path

        onnx_model = onnx.load_model(onnx_model_path)
        self.layers: dict[str, onnx.NodeProto] = {
            layer.name: layer for layer in onnx_model.graph.node
        }
        self.actual_tensors: list[str] = self.__init_tensors(onnx_model)

        pass

    def __init_tensors(self, model: onnx.ModelProto = None) -> list[str]:
        tensors = set()
        for layer in model.graph.node:
            tensors.update(layer.output)

        for tensor in model.graph.input:
            tensors.add(tensor.name)

        return list(tensors)
        pass

    def divide_layer(self, layer_name: str) -> onnx.ModelProto:

        layer = self.layers[layer_name]

        sub_input = []
        for input in layer.input:
            if input in self.actual_tensors:
                sub_input.append(input)

        sub_output = []
        for output in layer.output:
            sub_output.append(output)

        if self.quantized:
            quantized_inputs = []
            quantized_outputs = []
            # for input_name in sub_input:
            #     quantized_inputs.append(input_name + "_QuantizeLinear_Output")
            for output_name in sub_output:
                quantized_outputs.append(output_name + "_QuantizeLinear_Output")

            has_quantized_io = True
            # for input_name in quantized_inputs:
            #     if input_name not in self.quantized_tensors:
            #         has_quantized_io = False
            for output_name in quantized_outputs:
                if output_name not in self.quantized_tensors:
                    has_quantized_io = False

            if has_quantized_io:
                # sub_input = quantized_inputs
                sub_output = quantized_outputs

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
            onnx.utils.extract_model(
                self.model_path,
                temp_file.name,
                input_names=sub_input,
                output_names=sub_output,
            )
            sub_model = onnx.load_model(temp_file.name)
        return sub_model
