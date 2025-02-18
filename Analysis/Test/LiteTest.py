import time

import keras
import keras_cv
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
from Manipulation import Splitter, Unnester

NUM_RUN = 10


def runSignatureRunner(runner: SignatureRunner, input):
    liteInput = {}
    for key in runner.get_input_details().keys():
        liteInput[key] = input

    result = runner(**liteInput)
    return result


def build_converter(kerasModel: keras.Model):
    archive = keras.export.ExportArchive()
    archive.track(kerasModel)

    inputs = {}
    for key in kerasModel.input:
        inpTens = kerasModel.input[key]
        inputs[key] = tf.TensorSpec(
            shape=inpTens.shape, dtype=inpTens.dtype, name=inpTens.name
        )
    archive.add_endpoint(
        name="serve",
        fn=kerasModel.call,
        input_signature=[inputs],
    )
    archive.write_out(f"/tmp/saved_models/{kerasModel.name}", verbose=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(
        f"/tmp/saved_models/{kerasModel.name}"
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni TFLite
        tf.lite.OpsSet.SELECT_TF_OPS,  # Fallback ai kernel TensorFlow
    ]

    return converter


def setUpClass():
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")

    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=backbone,
        fpn_depth=2,
    )

    unnestedYolo: keras.Model = Unnester.unnestModel(yolo)
    unnestedYolo.save("./models/UnnestedYolo.keras")


def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 512, 512, 3)
        yield [data.astype(np.float32)]


def test_full_integer_quantization():
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")
    result_1 = kerasModel(images)

    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    interpreter = Interpreter(model_content=tflite_model)
    liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    result_2 = runSignatureRunner(liteRunner, images)

    diffNorm = tf.norm(result_1["box_0"] - result_2["box_0"], ord=np.inf)
    print(f"Full Integer Quant - Inf Norm of Difference >> {diffNorm}")


def divide_and_quantize(maxLayerNum: int, case: int):
    ## Splitting the Model
    yolo = keras.saving.load_model("./models/UnnestedYolo.keras")
    subModels = Splitter.modelSplit(yolo, maxLayerNum=maxLayerNum)

    tf_model_list = []
    for idx, subMod in enumerate(subModels):
        # subMod.save(f"/home/customuser/SubYolo_{idx}.keras")
        converter = build_converter(subMod)

        if case == 0:
            ## No Quantization
            pass

        elif case == 1:
            ## Dynamic Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            pass

        elif case == 2:
            ## Float16 Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            pass

        elif case == 3:
            ## Mized Quantization
            if idx % 2 == 0:
                ## Run With Dynamic Quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            else:
                ## Run With No Quantization
                pass
        else:
            raise Exception()

        tflite_model = converter.convert()
        tf_model_list.append(tflite_model)

    return tf_model_list


def division_run(tf_list: list):
    images = tf.ones(shape=(1, 512, 512, 3))

    ## Reading Sub Models only Once
    interpreterList = []
    for tf_model in tf_list:
        interpreterList.append(Interpreter(model_content=tf_model))

    startTime = time.time_ns()

    for _ in range(0, NUM_RUN):
        producedOutputs = {}

        producedOutputs["input_layer_1_0"] = images
        producedOutputs["input_layer_1_0_0"] = images

        for idx in range(0, len(tf_list)):
            interpreter = interpreterList[idx]
            runner: SignatureRunner = interpreter.get_signature_runner(
                "serving_default"
            )

            inputList = runner.get_input_details().keys()

            subInp = {}
            for inpName in inputList:
                subInp[inpName] = producedOutputs[inpName]
            subOut = runner(**subInp)
            subOut: dict
            for key in subOut.keys():
                out = subOut[key]
                producedOutputs[key] = out

    endTime = time.time_ns()

    yolo = keras.saving.load_model("./models/UnnestedYolo.keras")
    yoloOutput = yolo(images)

    diffNorm = tf.norm(yoloOutput["box_0"] - producedOutputs["box_0"], ord=np.inf)

    return [diffNorm, (endTime - startTime) / NUM_RUN]


def test_division_run(maxLayerNum):
    results = {}
    for x in range(0, 4):
        tf_list = divide_and_quantize(maxLayerNum=maxLayerNum, case=x)

        if x == 0:
            ## No Quantization
            print("No Quantization Info")
            pass

        elif x == 1:
            ## Dynamic Quantization
            print("Dynamic Quantization Info")
            pass

        elif x == 2:
            ## Float16 Quantization
            print("Float16 Quantization Info")
            pass

        elif x == 3:
            ## Mized Quantization
            print("Mixed Quantization Info")
        else:
            break

        run_info = division_run(tf_list)
        results[x] = run_info

    for x in results.keys():
        print(f"{x} >> {results[x]}")


if __name__ == "__main__":
    setUpClass()

    test_division_run(50_000)
