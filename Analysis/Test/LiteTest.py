import keras
import keras_cv
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
from Manipulation import Splitter, Unnester
import time

NUM_RUN = 25

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
    archive.write_out(f"/home/customuser/saved_models/{kerasModel.name}", verbose=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(
        f"/home/customuser/saved_models/{kerasModel.name}"
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


def test_yolo():
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")
    result_1 = kerasModel(images)

    converter = build_converter(kerasModel)
    tflite_model = converter.convert()
    interpreter = Interpreter(model_content=tflite_model)
    liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    
    result_2 = runSignatureRunner(liteRunner, images)

    diffNorm = tf.norm(result_1["box_0"] - result_2["box_0"], ord=np.inf)
    print(f"Yolo Test - Inf Norm of Difference >> {diffNorm}")


def test_mobile_net():
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel: keras.Model = keras.applications.MobileNetV3Large()
    result_1 = kerasModel(images)

    converter = build_converter(kerasModel)
    tflite_model = converter.convert()
    interpreter = Interpreter(model_content=tflite_model)
    liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    result_2 = runSignatureRunner(liteRunner, images)

    diffNorm = tf.norm(result_1 - result_2["output_0"], ord=np.inf)
    print(f"MobileNet Test - Inf Norm of Diff >> {diffNorm}")


def test_yolo_mixed():
    ## Splitting the Model
    yolo = keras.saving.load_model("./models/UnnestedYolo.keras")
    subModels = Splitter.modelSplit(yolo, maxLayerNum=50)
    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubYolo_{idx}.keras")

    ## Preparing for input/output management
    producedOutputs = {}
    images = tf.ones(shape=(1, 512, 512, 3))
    producedOutputs["input_layer_1_0"] = images
    producedOutputs["input_layer_1_0_0"] = images

    for idx in range(0, len(subModels)):
        print(f"Running Model {idx}")
        subMod: keras.Model = keras.saving.load_model(
            f"./models/SubYolo_{idx}.keras", compile=True
        )

        usingKeras = idx % 2 == 0

        if usingKeras:
            ## Run Keras Model
            inputList = subMod.input
            runner = subMod
        else:
            ## Run Lite Model
            converter = build_converter(subMod)
            tflite_model = converter.convert()
            interpreter = Interpreter(model_content=tflite_model)
            runner: SignatureRunner = interpreter.get_signature_runner(
                "serving_default"
            )

            inputList = runner.get_input_details().keys()

        subInp = {}
        for inpName in inputList:
            subInp[inpName] = producedOutputs[inpName]

        if usingKeras:
            subOut = runner(subInp)
        else:
            subOut = runner(**subInp)

        subOut: dict
        for key in subOut.keys():
            out = subOut[key]
            if isinstance(out, tf.Tensor):
                producedOutputs[key] = out.numpy()
            else:
                producedOutputs[key] = out

    yoloOutput = yolo(images)

    diffNorm = tf.norm(yoloOutput["box_0"] - producedOutputs["box_0"], ord=np.inf)
    print(f"Mixed Test - Inf Norm of Difference >> {diffNorm}")


def test_dynamic_quantization():
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")
    result_1 = kerasModel(images)

    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    interpreter = Interpreter(model_content=tflite_model)
    liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    result_2 = runSignatureRunner(liteRunner, images)

    diffNorm = tf.norm(result_1["box_0"] - result_2["box_0"], ord=np.inf)
    print(f"Dyn Quant - Inf Norm of Difference >> {diffNorm}")


def test_float16_quantization():
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")
    result_1 = kerasModel(images)

    converter = build_converter(kerasModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    interpreter = Interpreter(model_content=tflite_model)
    liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")
    result_2 = runSignatureRunner(liteRunner, images)

    diffNorm = tf.norm(result_1["box_0"] - result_2["box_0"], ord=np.inf)
    print(f"Float16 Quant - Inf Norm of Difference >> {diffNorm}")

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 512, 512, 3)
        yield [data.astype(np.float32)]

def test_full_integer_quantization() :
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


def test_mixed_quantization():
    ## Splitting the Model
    yolo = keras.saving.load_model("./models/UnnestedYolo.keras")
    subModels = Splitter.modelSplit(yolo, maxLayerNum=20)
    for idx, subMod in enumerate(subModels):
        subMod.save(f"/home/customuser/SubYolo_{idx}.keras")

    ## Preparing for input/output management
    producedOutputs = {}
    images = tf.ones(shape=(1, 512, 512, 3))
    producedOutputs["input_layer_1_0"] = images
    producedOutputs["input_layer_1_0_0"] = images

    for idx in range(0, len(subModels)):
        print(f"Running Model {idx}")
        subMod: keras.Model = keras.saving.load_model(
            f"/home/customuser/SubYolo_{idx}.keras", compile=True
        )
        converter = build_converter(subMod)

        if idx % 2 == 0:
            ## Run With No Quantization
            pass
        # elif idx % 3 == 1 :
        #     ## Run With Dyn Quantization
        #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            ## Run With Float16 Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
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

    yoloOutput = yolo(images)

    diffNorm = tf.norm(yoloOutput["box_0"] - producedOutputs["box_0"], ord=np.inf)
    print(f"Mixed Quantization Test - Inf Norm of Difference >> {diffNorm}")

def test_times() :
    images = tf.ones(shape=(1, 512, 512, 3))

    kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")
    for convIdx in range(0, 4) :
        converter = build_converter(kerasModel)
        if convIdx == 0 :
            ## None
            pass
        elif convIdx == 1 :
            ## Dynamic
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            pass
        elif convIdx == 2 :
            ## Float16
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            pass
        else :
            ## Full Integer
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            pass

        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        liteRunner: SignatureRunner = interpreter.get_signature_runner("serving_default")

        timesArray = np.zeros(shape = NUM_RUN)
        for runIdx in range(0, NUM_RUN) :
            start_time = time.time_ns()
            runSignatureRunner(liteRunner, images)
            end_time = time.time_ns()

            timesArray[runIdx] = end_time - start_time
        
        avgTime = timesArray.mean()
        if convIdx == 0 :
            print(f"No Quant Time >> {avgTime}")
        elif convIdx == 1 :
            print(f"Dynam Quant Time >> {avgTime}")
        elif convIdx == 2 :
            print(f"Float16 Quant Time >> {avgTime}")
        else :
            print(f"Full Int Quant Time >> {avgTime}")

if __name__ == "__main__":
    setUpClass()

    # test_mobile_net()
    # test_yolo()
    # test_dynamic_quantization()
    # test_float16_quantization()
    # test_full_integer_quantization()

    #test_mixed_quantization()

    test_times()
