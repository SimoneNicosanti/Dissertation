import pickle

import keras
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner


def check_precision():
    model: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")

    typesSet = set()
    for layer in model.layers:
        layer: keras.Layer = layer

        weightList: np.ndarray
        for weightList in layer.get_weights():
            typesSet.add(weightList.dtype)

    print(typesSet)


def general_convertion(kerasModel: keras.Model, outputPath: str) -> object:
    print("Converting Model")
    converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
    print("Built Converter")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni TFLite
        tf.lite.OpsSet.SELECT_TF_OPS,  # Fallback ai kernel TensorFlow
    ]
    tflite_model = converter.convert()
    print("Model Converted")

    # Save the model.
    if outputPath is not None:
        with open(outputPath, "wb") as f:
            f.write(tflite_model)

    return tflite_model


def general_comparison(
    kerasModel: keras.Model, liteModel: SignatureRunner, images
) -> tuple:

    liteInput = {}
    for key in liteModel.get_input_details().keys():
        liteInput[key] = images

    result_1 = kerasModel(images)
    result_2 = liteModel(**liteInput)

    return result_1, result_2


def testYolo():
    kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")
    general_convertion(kerasModel, "./models/UnnestedYolo.tflite")

    interpreter = Interpreter("./models/UnnestedYolo.tflite")
    interpreter.allocate_tensors()
    liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")

    images = tf.random.uniform(shape=(1, 512, 512, 3))
    result_1, result_2 = general_comparison(kerasModel, liteModel, images)

    compareResults(result_1["box_0"], result_2["box_0"], "Box")
    compareResults(result_1["class_0"], result_2["class_0"], "Class")


def compareResults(res_1, res_2, outName):
    print(f"Are Equal {outName} >>> {np.array_equal(res_1, res_2)}")
    print(f"Norm Of Difference {outName} >>> {tf.norm(res_1 - res_2)}")
    print()


def testMobileNet():
    kerasModel: keras.Model = keras.applications.MobileNetV3Large()
    general_convertion(kerasModel, "./models/Lite_MobileLarge.tflite")

    interpreter = Interpreter("./models/Lite_MobileLarge.tflite")
    interpreter.allocate_tensors()
    liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")

    result_1, result_2 = general_comparison(kerasModel, liteModel, readTestElem())

    compareResults(result_1, result_2["output_0"], "Output_0")
    print(f"Predictions >>> ({np.argmax(result_1)}, {np.argmax(result_2['output_0'])})")


def testSubYolo():
    model: keras.Model = keras.saving.load_model("./models/SubYolo_1.keras")
    general_convertion(model, "./models/Lite_SubYolo_1.tflite")

    interpreter = Interpreter("./models/Lite_SubYolo_1.tflite")
    liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")
    print("Input Signature >>> ", liteModel.get_input_details())
    print("Output Signature >>> ", liteModel.get_output_details())


def testMixed_1():
    for idx in range(0, 9):
        keras.backend.clear_session()
        print(f"Running Model {idx}")
        # subMod: keras.Model = keras.saving.load_model(
        #     f"./models/SubYolo_{idx}.keras", compile=False
        # )

        loaded = tf.saved_model.load(f"./models/SubYolo_{idx}")
        print(loaded.signatures)
        print()


def testMixed():
    producedOutputs = {}
    images = tf.random.uniform(shape=(1, 512, 512, 3))

    producedOutputs["input_layer_1_0"] = images
    producedOutputs["input_layer_1_0_0"] = images
    for idx in range(0, 9):
        print(f"Running Model {idx}")
        subMod: keras.Model = keras.saving.load_model(
            f"./models/SubYolo_{idx}.keras", compile=False
        )
        randInt: int = np.random.randint(0, 2)
        if randInt == 0:
            print("Running with Keras")
            ## Keras Exec
            subInp = {}
            for inpName in subMod.input:
                subInp[inpName] = producedOutputs[inpName]

            subOut = subMod(subInp)
        else:
            print("Running with LiteRT")
            ## Lite Exec
            liteModContent = general_convertion(subMod, None)
            interpreter = Interpreter(model_content=liteModContent)
            interpreter.allocate_tensors()
            liteModel: SignatureRunner = interpreter.get_signature_runner(
                "serving_default"
            )

            subInp = {}
            for inpName in liteModel.get_input_details():
                subInp[inpName] = producedOutputs[inpName]

            subOut = liteModel(**subInp)

        for outName in subOut:
            producedOutputs[outName] = subOut[outName]

    wholeModel = keras.saving.load_model("./models/UnnestedYolo.keras")
    wholeModelOutput = wholeModel(images)

    print()
    compareResults(wholeModelOutput["box_0"], producedOutputs["box_0"], "Box")
    compareResults(wholeModelOutput["class_0"], producedOutputs["class_0"], "Class")


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem, dtype=tf.float32)


if __name__ == "__main__":
    # testYolo()
    # testMobileNet()
    # testSubYolo()
    testMixed()
    # testMixed_1()
