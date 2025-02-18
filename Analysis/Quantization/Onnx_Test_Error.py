import numpy as np
import onnxruntime
import tensorflow as tf
import tf_keras
from onnxruntime.quantization.quantize import QuantFormat, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_NAME = "./models/ResNet152"


def generate_input():
    return tf.zeros(shape=(1, 224, 224, 3))


def test_keras():
    model: tf_keras.Model = tf_keras.applications.ResNet152V2(input_shape=(224, 224, 3))
    return model(generate_input())


def test_onnx():
    sess = onnxruntime.InferenceSession(MODEL_NAME + ".onnx")
    return sess.run(None, input_feed={"args_0": generate_input().numpy()})


def test_onnx_quant():
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

    sess = onnxruntime.InferenceSession(MODEL_NAME + "_quant.onnx")
    return sess.run(None, input_feed={"args_0": generate_input().numpy()})


def main():
    base_out = test_keras()
    print(np.argmax(base_out[0]))

    onnx_out = test_onnx()
    print("Error Keras VS Onnx >> ", np.linalg.norm(base_out - onnx_out[0], ord=np.inf))
    print(np.argmax(onnx_out[0][0]))

    quant_onnx_out = test_onnx_quant()
    print(
        "Error Onnx VS Onnx Quant >> ",
        np.linalg.norm(onnx_out[0] - quant_onnx_out[0], ord=np.inf),
    )
    print(np.argmax(quant_onnx_out[0][0]))

    pass


if __name__ == "__main__":
    main()
