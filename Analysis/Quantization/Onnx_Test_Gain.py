import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import ModelProto
from onnxruntime.quantization.quantize import (
    QuantFormat,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

NUM_RUNS = 10
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
        if self.idx == 1:
            return None
        inp = generate_input(self.model)
        self.idx += 1

        return inp


def quantize_model(subModelName, nodesToQuantize=None):
    quant_pre_process(
        subModelName + ".onnx", output_model_path=subModelName + "_pre_quant.onnx"
    )
    quantize_static(
        model_input=subModelName + "_pre_quant.onnx",
        model_output=subModelName + "_quant.onnx",
        quant_format=QuantFormat.QOperator,
        calibration_data_reader=MyDataReader(subModelName + "_pre_quant.onnx"),
        nodes_to_quantize=nodesToQuantize,
    )


def compute_avg_time(sess_options, modelName):
    model = onnx.load_model(modelName)
    sess = ort.InferenceSession(modelName, providers=["OpenVINOExecutionProvider"])

    input = generate_input(model)
    start = time.time_ns()
    for _ in range(0, NUM_RUNS):
        out = sess.run(None, input_feed=input)
    end = time.time_ns()
    return (end - start) / (NUM_RUNS * 1e6), out


def main():
    onnx_model: ModelProto = onnx.load_model(MODEL_NAME + ".onnx")

    actual_outputs = set()
    actual_outputs = actual_outputs.union(list(onnx_model.graph.node[0].input))
    for node in onnx_model.graph.node:
        actual_outputs = actual_outputs.union(list(node.output))

    times_dict = {}

    idx = 0
    for node in onnx_model.graph.node:
        print(f"{idx}/{len(onnx_model.graph.node)}")

        sub_input = [inp for inp in node.input if inp in actual_outputs]
        onnx.utils.extract_model(
            MODEL_NAME + ".onnx", MODEL_NAME + "_sub.onnx", sub_input, node.output
        )
        not_quantized_time, not_quant_out = compute_avg_time(
            ort.SessionOptions(), MODEL_NAME + "_sub.onnx"
        )

        quantize_model(MODEL_NAME + "_sub")
        quantized_time, quant_out = compute_avg_time(
            ort.SessionOptions(), MODEL_NAME + "_sub_quant.onnx"
        )

        times_dict[(node.name, node.op_type)] = [
            not_quantized_time,
            quantized_time,
            not_quantized_time / quantized_time,
            np.linalg.norm(not_quant_out[0] - quant_out[0]),
        ]

        idx += 1

    tot_times = [0, 0]
    for key, value in times_dict.items():
        print(key, value)
        tot_times[0] += value[0]
        tot_times[1] += value[1]

    print(tot_times[0], tot_times[1], tot_times[0] / tot_times[1])

    quantizeNodes = []
    for key, value in times_dict.items():
        if value[0] > value[1]:
            quantizeNodes.append(key[0])

    print(compute_avg_time(ort.SessionOptions(), MODEL_NAME + ".onnx")[0])

    quantize_model(MODEL_NAME, quantizeNodes)
    print(compute_avg_time(ort.SessionOptions(), MODEL_NAME + "_quant.onnx")[0])

    quantize_model(MODEL_NAME)
    print(compute_avg_time(ort.SessionOptions(), MODEL_NAME + ".onnx")[0])


if __name__ == "__main__":
    main()
