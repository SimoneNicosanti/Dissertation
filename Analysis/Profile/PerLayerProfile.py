import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

TEST_NUM = 3
MODEL_NAME = "yolo11l"


def collect_valid_tensors(onnx_model: onnx.ModelProto):

    tensor_names = []

    for layer in onnx_model.graph.node:
        for output in layer.output:
            tensor_names.append(output)

    input_names = [inp.name for inp in onnx_model.graph.input]
    tensor_names.extend(input_names)

    return tensor_names


def profile_model(sub_model: onnx.ModelProto):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess = ort.InferenceSession(
        sub_model.SerializeToString(), sess_options=sess_options
    )

    input_dict = {}
    for input in sess.get_inputs():
        input_dict[input.name] = np.zeros(input.shape, dtype=np.float32)

    for _ in range(TEST_NUM):
        sess.run(None, input_feed=input_dict)

    profile_file = sess.end_profiling()

    with open(profile_file, "r") as profile:
        json_array = json.load(profile)
        filtered_array = filter(
            lambda elem: elem["cat"] == "Session"
            and elem["name"] == "SequentialExecutor::Execute",
            json_array,
        )
        tot = sum(elem["dur"] for elem in filtered_array)

    os.remove(profile_file)
    return (tot * 1e-6) / TEST_NUM

    pass


def divide_layer(
    model_path: str,
    layer_name: str,
    onnx_model: onnx.ModelProto,
    valid_tensors: list[str],
):

    sub_mod_inp = []
    sub_mod_out = []

    layer_info = None
    for layer in onnx_model.graph.node:
        if layer.name == layer_name:
            layer_info = layer
            break

    for inp in layer_info.input:
        if inp in valid_tensors:
            sub_mod_inp.append(inp)

    for out in layer_info.output:
        if out in valid_tensors:
            sub_mod_out.append(out)

    extractor = Extractor(onnx_model)
    extracted_model = extractor.extract_model(sub_mod_inp, sub_mod_out)

    return extracted_model


def main():
    onnx_model = onnx.load_model(MODEL_NAME + ".onnx")
    valid_tensors = collect_valid_tensors(onnx_model)

    times = {}
    for layer in onnx_model.graph.node:
        print("Processing >> ", layer.name)
        extraxcted_model = divide_layer(
            MODEL_NAME + ".onnx", layer.name, onnx_model, valid_tensors
        )
        layer_time = profile_model(extraxcted_model)
        times[layer.name] = layer_time

    pass


if __name__ == "__main__":
    main()
