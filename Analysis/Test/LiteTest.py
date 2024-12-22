import pickle

import keras
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import unittest
from Manipulation import Unnester, Splitter
import keras_cv




def testSubYolo():
    model: keras.Model = keras.saving.load_model("./models/SubYolo_1.keras")
    general_convertion(model, "./models/Lite_SubYolo_1.tflite")

    interpreter = Interpreter("./models/Lite_SubYolo_1.tflite")
    liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")
    print("Input Signature >>> ", liteModel.get_input_details())
    print("Output Signature >>> ", liteModel.get_output_details())


def saved_model_convertion(kerasModel: keras.Model):

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
    archive.write_out(f"/tmp/{kerasModel.name}")

    converter = tf.lite.TFLiteConverter.from_saved_model(f"/tmp/{kerasModel.name}")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni TFLite
        tf.lite.OpsSet.SELECT_TF_OPS,  # Fallback ai kernel TensorFlow
    ]
    tflite_model = converter.convert()
    with open(f"/tmp/{kerasModel.name}.tflite", "wb") as f:
        f.write(tflite_model)

    return tflite_model

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

def executeBoth(
    kerasModel: keras.Model, liteModel: SignatureRunner, images
) -> tuple:

    liteInput = {}
    for key in liteModel.get_input_details().keys():
        liteInput[key] = images

    result_1 = kerasModel(images)
    result_2 = liteModel(**liteInput)

    return result_1, result_2




class LiteTest(unittest.TestCase) :

    @classmethod
    def setUp_Yolo(cls) :
        backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")

        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=backbone,
            fpn_depth=2,
        )

        unnestedYolo : keras.Model = Unnester.unnestModel(yolo)
        unnestedYolo.save("./models/UnnestedYolo.keras")

        kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")
        general_convertion(kerasModel, "/tmp/UnnestedYolo.tflite")

        subModels = Splitter.modelSplit(kerasModel, maxLayerNum=50)
        for idx, subMod in enumerate(subModels) :
            subMod.save(f"./models/SubYolo_{idx}.keras")
            saved_model_convertion(subMod)


    @classmethod
    def setUp_MobileNet(cls) :
        kerasModel: keras.Model = keras.applications.MobileNetV3Large()
        general_convertion(kerasModel, "/tmp/Lite_MobileLarge.tflite")

    
    @classmethod
    def setUpClass(cls):
        cls.setUp_Yolo()
        cls.setUp_MobileNet()


    def test_yolo(self) :
        kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")

        interpreter = Interpreter("/tmp/UnnestedYolo.tflite")
        interpreter.allocate_tensors()
        liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")

        images = tf.ones(shape=(1, 512, 512, 3))
        result_1, result_2 = executeBoth(kerasModel, liteModel, images)

        diffNorm = tf.norm(result_1["box_0"] - result_2["box_0"])
        self.assertAlmostEqual(diffNorm, 0, delta = 1.e-3)


    def test_MobileNet(self):
        kerasModel: keras.Model = keras.applications.MobileNetV3Large()

        interpreter = Interpreter("/tmp/Lite_MobileLarge.tflite")
        interpreter.allocate_tensors()
        liteModel: SignatureRunner = interpreter.get_signature_runner("serving_default")

        images = tf.ones(shape = (1, 512, 512, 3))

        result_1, result_2 = executeBoth(kerasModel, liteModel, images)
        
        diffNorm = tf.norm(result_1 - result_2["output_0"])
        self.assertAlmostEqual(diffNorm, 0, delta = 1.e-3)


    def test_mixed(self):
        producedOutputs = {}
        images = tf.ones(shape=(1, 512, 512, 3))

        producedOutputs["input_layer_1_0"] = images
        producedOutputs["input_layer_1_0_0"] = images
        for idx in range(0, 9):
            print(f"Running Model {idx}")
            subMod: keras.Model = keras.saving.load_model(
                f"./models/SubYolo_{idx}.keras", compile=True
            )
            #randInt: int = np.random.randint(0, 2)
            if idx % 2 == 1:
                print("Running with Keras")
                ## Keras Exec
                subInp = {}
                for inpName in subMod.input:
                    subInp[inpName] = producedOutputs[inpName]

                subOut: dict[str, tf.Tensor] = subMod(subInp)
                for key in subOut.keys():
                    subOut[key] = subOut[key].numpy()
            else:
                print("Running with LiteRT")
                ## Lite Exec
                interpreter = Interpreter(model_path=f"/tmp/{subMod.name}.tflite")
                interpreter.allocate_tensors()
                liteModel: SignatureRunner = interpreter.get_signature_runner(
                    "serving_default"
                )
                subInp = {}
                for inpName in liteModel.get_input_details():
                    subInp[inpName] = producedOutputs[inpName]

                subOut: dict[str, np.ndarray] = liteModel(**subInp)

            for outName in subOut:
                producedOutputs[outName] = subOut[outName]

        yolo = keras.saving.load_model("./models/UnnestedYolo.keras")
        wholeModelOutput = yolo(images)

        diffNorm = tf.norm(wholeModelOutput["box_0"] - producedOutputs["box_0"])
        self.assertAlmostEqual(diffNorm, 0, delta = 1.e-3)


if __name__ == "__main__":
    unittest.main()