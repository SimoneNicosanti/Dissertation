import time

import keras
import numpy as np
import tensorflow as tf
import tflite
from ai_edge_litert.interpreter import Interpreter, SignatureRunner


def runSignatureRunner(runner: SignatureRunner, input):
    liteInput = {}
    for key in runner.get_input_details().keys():
        liteInput[key] = input

    result = runner(**liteInput)
    return result


def build_converter(kerasModel: keras.Model):

    converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter


def representative_dataset_gen():
    yield [tf.ones(shape=(1, 224, 224, 3), dtype=tf.float32)]


def prepare_model():
    converter = build_converter(keras.applications.ResNet50())
    tf_model = converter.convert()
    return tf_model


def main():
    tf_model = prepare_model()

    model: tflite.Model = tflite.Model.GetRootAs(tf_model)
    subgraph = model.Subgraphs(0)  # Typically, models have one subgraph

    # Extract tensor details
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode("utf-8")
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        dtype = tensor.Type()

        print(f"Tensor {i}: {name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {dtype}")
        print("-" * 40)

    with open("./models/ResNet50_quant.tflite", "wb") as f:
        f.write(tf_model)
    # compute_avg_time(tf_model)


if __name__ == "__main__":
    main()
