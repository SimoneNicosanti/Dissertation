import time

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import tf_keras
from onnxruntime.quantization.quantize import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

NUM_RUNS = 100
MODEL_NAME = "./models/ResNet152"


def prepare_general_test():
    model = tf_keras.applications.ResNet152V2()
    model.save(MODEL_NAME + ".keras", save_format="keras")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
    with open(MODEL_NAME + ".onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def prepare_quantized_test(
    quant_format: QuantFormat, activation_type: QuantType, weight_type: QuantType
):
    class MyDataReader:
        def __init__(self):
            self.idx = 0

        def get_next(self):
            self.idx += 1
            if self.idx == 50:
                return None

            return {"args_0": np.zeros((1, 224, 224, 3), dtype=np.float32)}

    quant_pre_process(
        MODEL_NAME + ".onnx",
        output_model_path=MODEL_NAME + "_pre_quant.onnx",
    )
    quantize_static(
        model_input=MODEL_NAME + "_pre_quant.onnx",
        model_output=MODEL_NAME + "_quant.onnx",
        quant_format=quant_format,
        calibration_data_reader=MyDataReader(),
        activation_type=activation_type,
        weight_type=weight_type,
    )


def not_quantized_test():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    avg_time = compute_avg_time(sess_options, MODEL_NAME + ".onnx")
    print("ONNX Not Quantized Time >> ", avg_time)


def compute_avg_time(sess_options, modelName):
    if "CUDAExecutionProvider" in ort.get_available_providers():
        sess = ort.InferenceSession(
            modelName, sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
    else:
        sess = ort.InferenceSession(modelName, sess_options=sess_options)

    time_array = np.zeros(NUM_RUNS)

    ## Warmup Run
    sess.run(None, input_feed={"args_0": np.zeros((1, 224, 224, 3), dtype=np.float32)})
    for i in range(0, NUM_RUNS):
        start = time.perf_counter_ns()
        sess.run(
            None, input_feed={"args_0": np.zeros((1, 224, 224, 3), dtype=np.float32)}
        )
        end = time.perf_counter_ns()
        time_array[i] = (end - start) / 1e6

        if i == 0:
            print("First Run Time >> ", time_array[i])

    return time_array.mean()


def quantized_test(test_case: str):

    sess_options = ort.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.optimized_model_filepath = MODEL_NAME + "_quant_opt.onnx"

    avg_time = compute_avg_time(sess_options, MODEL_NAME + "_quant.onnx")
    print(f"{test_case} >> {avg_time}")


def main():
    prepare_general_test()
    not_quantized_test()

    test_cases = [
        (QuantFormat.QDQ, QuantType.QInt8, QuantType.QInt8),
        (QuantFormat.QDQ, QuantType.QUInt8, QuantType.QInt8),
        (QuantFormat.QDQ, QuantType.QUInt8, QuantType.QUInt8),
        (QuantFormat.QOperator, QuantType.QInt8, QuantType.QInt8),
        (QuantFormat.QOperator, QuantType.QUInt8, QuantType.QInt8),
        (QuantFormat.QOperator, QuantType.QUInt8, QuantType.QUInt8),
    ]

    for test_case in test_cases:
        prepare_quantized_test(
            quant_format=test_case[0],
            activation_type=test_case[1],
            weight_type=test_case[2],
        )
        quantized_test(
            f"Quantized ({test_case[0].name}, Activation {test_case[1].name}, Weight {test_case[2].name})"
        )


if __name__ == "__main__":
    main()
