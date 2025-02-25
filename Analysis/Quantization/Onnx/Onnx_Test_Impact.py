import numpy as np
import onnx
import onnxruntime
from onnx import ModelProto
from onnxruntime.quantization.quantize import (
    QuantFormat,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_NAME = "./models/ResNet50"


def generate_input(model: ModelProto):
    # Get input shape from the model
    input_dict = {}
    for input_tensor in model.graph.input:
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_shape[0] = 1

        # Generate a random input matching the model's input type
        random_input = np.random.rand(*input_shape).astype(np.float32)
        input_dict[input_tensor.name] = random_input

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
        quant_format=QuantFormat.QDQ,
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
    input = {"args_0": np.random.rand(1, 224, 224, 3).astype(np.float32)}
    onnx_model = onnx.load(MODEL_NAME + "_pre_quant.onnx")

    i = 0
    tot = 0
    node_list = []
    for node in onnx_model.graph.node:
        if i == 50:
            break
        node_list.append(node.name)
        quantize(node_list)

        diff_norm = test_difference(input)
        i += 1
        print(f"Tot Nodes {i} >> {diff_norm}")
        tot += diff_norm
    print(tot / 50)

    pass


if __name__ == "__main__":
    main()
