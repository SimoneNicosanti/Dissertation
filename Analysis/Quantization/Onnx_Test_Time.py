import time

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import tf_keras
from onnxruntime.quantization.quantize import (
    QuantFormat,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

NUM_RUNS = 250
MODEL_NAME = "./models/ResNet152"


def prepare_general_test():
    model = tf_keras.applications.ResNet152V2()
    model.save(MODEL_NAME + ".keras", save_format="keras")
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

    sess_options = ort.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.optimized_model_filepath = MODEL_NAME + "_quant_opt.onnx"

    avg_time = compute_avg_time(sess_options, MODEL_NAME + "_quant.onnx")
    print("ONNX Int8 Quantized Time >> ", avg_time)


def main():
    prepare_general_test()
    prepare_quantized_test()

    not_quantized_test()
    quantized_test()


if __name__ == "__main__":
    main()
