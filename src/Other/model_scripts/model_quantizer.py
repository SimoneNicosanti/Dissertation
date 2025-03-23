import tempfile

import numpy
import onnx
from onnxruntime.quantization.quantize import (
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class MyDataReader:
    def __init__(self, modelFileName: str):
        self.idx = 0
        self.model = onnx.load_model(modelFileName)

    def get_next(self):
        if self.idx == 1:
            return None

        ## Just to quantize, not to get good quantization results...
        input_dict = {}
        for input in self.model.graph.input:
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            input_dict[input.name] = numpy.zeros(shape=shape, dtype=numpy.float32)
        self.idx += 1

        return input_dict


def quantize(model_name, exclude_nodes):
    model_path = "../models/" + model_name + ".onnx"
    temp_mod_path = tempfile.mktemp()
    quant_pre_process(input_model=model_path, output_model_path=temp_mod_path)

    quant_model_path = "../models/" + model_name + "_quant.onnx"
    quantize_static(
        model_input=temp_mod_path,
        model_output=quant_model_path,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_data_reader=MyDataReader(model_path),
        nodes_to_exclude=exclude_nodes,
    )


def main():
    model_list = [
        "yolo11n-seg",
        "yolo11s-seg",
        "yolo11l-seg",
        "yolo11x-seg",
        "yolo11n",
        "yolo11s",
        "yolo11l",
        "yolo11x",
    ]

    for model in model_list:
        print("Quantizing >> ", model)
        model_path = "../models/" + model + ".onnx"
        onnx_model = onnx.load_model(model_path)

        exclude_nodes = []
        for node in onnx_model.graph.node:
            if node.op_type == "Concat":
                exclude_nodes.append(node.name)

        quantize(model, exclude_nodes)

    pass


if __name__ == "__main__":
    main()
