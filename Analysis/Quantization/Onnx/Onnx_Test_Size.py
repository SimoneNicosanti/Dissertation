import time

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import tf_keras
from onnx import ValueInfoProto
from onnxruntime.quantization.quantize import QuantFormat, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

NUM_RUNS = 250
MODEL_NAME = "./models/ResNet50"


def prepare_general_test():
    model = tf_keras.applications.ResNet50()
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
    with open(MODEL_NAME + ".onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def prepare_quantized_test():
    class MyDataReader:
        def __init__(self):
            self.idx = 0

        def get_next(self):
            self.idx += 1
            if self.idx == 50:
                return None

            return {"args_0": np.zeros((1, 224, 224, 3), dtype=np.float32)}

    quant_pre_process(
        MODEL_NAME + ".onnx", output_model_path=MODEL_NAME + "_pre_quant.onnx"
    )
    quantize_static(
        model_input=MODEL_NAME + "_pre_quant.onnx",
        model_output=MODEL_NAME + "_quant.onnx",
        quant_format=QuantFormat.QDQ,
        calibration_data_reader=MyDataReader(),
    )


def not_quantized_test():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    avg_time = compute_avg_time(sess_options, MODEL_NAME + ".onnx")
    print("ONNX Not Quantized Time >> ", avg_time)


def compute_avg_time(sess_options, modelName):
    print(ort.get_available_providers())
    if "CUDAExecutionProvider" in ort.get_available_providers():
        sess = ort.InferenceSession(
            modelName, sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
    else:
        sess = ort.InferenceSession(modelName, sess_options=sess_options)

    start = time.time_ns()
    for _ in range(0, NUM_RUNS):
        sess.run(
            None, input_feed={"args_0": np.zeros((1, 224, 224, 3), dtype=np.float32)}
        )
    end = time.time_ns()
    return (end - start) / (NUM_RUNS * 1e6)


def quantized_test():
    onnxModel: onnx.ModelProto = onnx.load_model(MODEL_NAME + "_quant.onnx")
    infered = onnx.shape_inference.infer_shapes(onnxModel, data_prop=True)

    elems = []
    info: ValueInfoProto
    for info in infered.graph.value_info:
        print(info)
        if info.type.tensor_type.elem_type == 3:
            elems.append(info)
    print(elems)
    print(len(elems))
    print(len(infered.graph.value_info))

    baseModel: onnx.ModelProto = onnx.load_model(MODEL_NAME + ".onnx")
    baseInfered = onnx.shape_inference.infer_shapes(baseModel, data_prop=True)
    print(len(baseInfered.graph.value_info))


def main():
    # prepare_general_test()
    # prepare_quantized_test()

    # not_quantized_test()
    quantized_test()


if __name__ == "__main__":
    main()
