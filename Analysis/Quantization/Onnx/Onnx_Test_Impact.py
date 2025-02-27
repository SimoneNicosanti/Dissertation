import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime
from onnx import ModelProto, version_converter
from onnx.numpy_helper import to_array
from onnxruntime.quantization.quantize import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_NAME = "../models/resnet50-v1-7/resnet50-v1-7"


def generate_input(model: ModelProto):
    # Get input shape from the model
    input_dict = {}

    input_dict["data"] = to_array(
        onnx.load_tensor("../models/resnet50-v1-7/test_data_set_0/input_0.pb")
    )

    return input_dict


class MyDataReader:
    def __init__(self, modelFileName: str):
        self.idx = 0
        self.model = onnx.load_model(modelFileName)

    def get_next(self):
        if self.idx == 10:
            return None
        inp = generate_input(self.model)
        self.idx += 1

        return inp


def quantize(name_list=None):
    quantize_static(
        model_input=MODEL_NAME + "_pre_quant.onnx",
        model_output=MODEL_NAME + "_quant.onnx",
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_data_reader=MyDataReader(MODEL_NAME + "_pre_quant.onnx"),
        nodes_to_quantize=name_list,
    )


def test_difference(input):
    sess = onnxruntime.InferenceSession(MODEL_NAME + "_quant.onnx")

    quantized_result = sess.run(None, input_feed=input)

    sess = onnxruntime.InferenceSession(MODEL_NAME + "_pre_quant.onnx")

    original_result = sess.run(None, input_feed=input)

    return np.linalg.norm(quantized_result[0] - original_result[0])


def main():
    prev_model = onnx.load(MODEL_NAME + ".onnx")
    new_model = version_converter.convert_version(prev_model, 20)
    onnx.save_model(new_model, MODEL_NAME + ".onnx")

    quant_pre_process(
        MODEL_NAME + ".onnx",
        output_model_path=MODEL_NAME + "_pre_quant.onnx",
    )

    onnx_model = onnx.load(MODEL_NAME + ".onnx")
    input = generate_input(onnx_model)
    node_list = []
    norm_list = []
    for idx, node in enumerate(onnx_model.graph.node):

        node_list.append(node.name)
        quantize([node.name])

        diff_norm = test_difference(input)
        norm_list.append(diff_norm)
        print(f"{idx}/{len(onnx_model.graph.node)} >> {diff_norm}")

    plt.plot(norm_list)
    plt.show()
    pass


if __name__ == "__main__":
    main()
