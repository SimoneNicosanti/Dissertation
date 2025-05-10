import onnx


class LayerDivider:

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model: onnx.ModelProto = onnx_model
        self.extractor: onnx.utils.Extractor = onnx.utils.Extractor(onnx_model)

        self.layers: dict[str, onnx.NodeProto] = {
            layer.name: layer for layer in onnx_model.graph.node
        }
        self.actual_tensors: list[str] = self.__init_tensors()

        pass

    def __init_tensors(self) -> list[str]:
        tensors = set()
        for layer in self.onnx_model.graph.node:
            tensors.update(layer.output)

        for tensor in self.onnx_model.graph.input:
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

        sub_model: onnx.ModelProto = self.extractor.extract_model(sub_input, sub_output)

        return sub_model
