import csv

import numpy as np
import onnx
import onnxruntime
import skopt
from onnx import ModelProto
from onnxruntime.quantization.quantize import (
    QuantFormat,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_NAME = "../models/ResNet50"

random_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
random_calibration = np.random.rand(10, 224, 224, 3).astype(np.float32)
onnx_model = onnx.load(MODEL_NAME + "_pre_quant.onnx")


def prepare_quantization(names=None):
    class MyDataReader:
        def __init__(self):
            self.idx = 0

        def get_next(self):
            calibration_elem = random_calibration[self.idx]
            self.idx += 1
            if self.idx == 10:
                return None

            return {"args_0": [calibration_elem]}

    quantize_static(
        model_input=MODEL_NAME + "_pre_quant.onnx",
        model_output=MODEL_NAME + "_quant.onnx",
        quant_format=QuantFormat.QDQ,
        calibration_data_reader=MyDataReader(),
        nodes_to_quantize=names,
    )


def objective(quantized_names):

    names = []
    speedup = 0
    max_speedup = sum(speedup_dict.values())
    for idx, node in enumerate(onnx_model.graph.node):
        if quantized_names[idx] == 1.0:
            names.append(node.name)
            speedup += speedup_dict[node.name]
    print("Optimizing with >> ", sum(quantized_names))

    prepare_quantization(names)

    sess = onnxruntime.InferenceSession(MODEL_NAME + "_quant.onnx")
    quant_res = sess.run(None, input_feed={"args_0": random_calibration})

    sess = onnxruntime.InferenceSession(MODEL_NAME + "_pre_quant.onnx")
    not_quant_res = sess.run(None, input_feed={"args_0": random_calibration})

    error = np.linalg.norm(quant_res[0] - not_quant_res[0])
    norm_error = error / np.linalg.norm(not_quant_res[0])
    norm_speedup = speedup / max_speedup
    print("\tError >> ", error)
    print("\tValue >> ", norm_error - norm_speedup)
    return norm_error - norm_speedup


speedup_dict = {}


def main():

    model = onnx.load(MODEL_NAME + "_pre_quant.onnx")

    model_node_names = [node.name for node in model.graph.node]
    list_nodes_vars = []
    with open("./SpeedUp.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] not in model_node_names:
                print(row[0])

            if float(row[1]) > 1.0:
                list_nodes_vars.append(skopt.space.Categorical([0, 1]))
                speedup_dict[row[0]] = float(row[1])
            else:
                list_nodes_vars.append(skopt.space.Categorical([0]))

    res = skopt.gp_minimize(
        objective,  # the function to minimize
        list_nodes_vars,  # the bounds on each dimension of x
        acq_func="EI",  # the acquisition function
        n_calls=75,  # the number of evaluations of f
        n_random_starts=10,  # the number of random initialization points
        random_state=1234,
    )

    for idx, node in enumerate(model.graph.node):
        if res.x[idx] == 1.0:
            print(node.name)
    print("Final Value >> ", res.fun)


if __name__ == "__main__":
    main()
