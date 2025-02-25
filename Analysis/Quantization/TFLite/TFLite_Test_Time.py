import time

import keras
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner

NUM_RUN = 1


def runSignatureRunner(runner: SignatureRunner, input):
    liteInput = {}
    for key in runner.get_input_details().keys():
        liteInput[key] = input

    result = runner(**liteInput)
    return result


def build_converter(kerasModel: keras.Model):

    converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni TFLite
        tf.lite.OpsSet.SELECT_TF_OPS,  # Fallback ai kernel TensorFlow
    ]
    return converter


def test_no_quantization():
    kerasModel = keras.applications.ResNet152V2()
    converter = build_converter(kerasModel)

    tf_model = converter.convert()
    return compute_avg_time(tf_model)


def test_dynamic():
    kerasModel = keras.applications.ResNet152V2()
    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tf_model = converter.convert()
    return compute_avg_time(tf_model)


def test_float16():
    kerasModel = keras.applications.ResNet152V2()
    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tf_model = converter.convert()
    return compute_avg_time(tf_model)

    pass


def test_int8_fallback():
    def representative_dataset_gen():
        yield [tf.ones(shape=(1, 224, 224, 3), dtype=tf.float32)]

    kerasModel = keras.applications.ResNet152V2()
    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    tf_model = converter.convert()
    return compute_avg_time(tf_model)


def test_int8_full():
    def representative_dataset_gen():
        yield [tf.ones(shape=(1, 224, 224, 3), dtype=tf.float32)]

    kerasModel = keras.applications.ResNet152V2()
    converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tf_model = converter.convert()
    return compute_avg_time(tf_model, tf.uint8)


def compute_avg_time(tf_model, dtype=tf.float32):
    input = tf.ones(shape=(1, 224, 224, 3), dtype=dtype)
    # input = np.expand_dims(images[0], axis=0).astype(dtype)
    ## Reading Sub Models only Once
    interpreter = Interpreter(model_content=tf_model)

    runner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    startTime = time.perf_counter_ns()

    for _ in range(0, 1):
        runSignatureRunner(runner, input)

    endTime = time.perf_counter_ns()

    return (endTime - startTime) / (1 * 1e6)


def main():
    no_quant_time = test_no_quantization()
    dyn_quant_time = test_dynamic()
    float_quant_time = test_float16()
    int_quant_time = test_int8_fallback()
    full_int_quant_time = test_int8_full()

    print("No Quant Time >> ", no_quant_time)
    print("Dyn Quant Time >> ", dyn_quant_time)
    print("Float16 Quant Time >> ", float_quant_time)
    print("Int8 Fallback Quant Time >> ", int_quant_time)
    print("Int8 Full Quant Time >> ", full_int_quant_time)


if __name__ == "__main__":
    main()
