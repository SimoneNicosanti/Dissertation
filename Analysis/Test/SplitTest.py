import keras
import keras_cv
import numpy as np
import tensorflow as tf
from Manipulation import Splitter, Unnester


def setUpTest():
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

    yolo: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")

    wholeModelOutput = yolo(images)

    subModels: list[keras.Model] = Splitter.modelSplit(yolo, maxLayerNum=50)
    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubYolo_{idx}.keras")

    loadedModels = []
    for idx in range(0, 9):
        loadedModels.append(keras.saving.load_model(f"./models/SubYolo_{idx}.keras"))

    producedOutputs: dict[str] = {}
    producedOutputs["input_layer_1_0_0"] = images

    for idx, subMod in enumerate(loadedModels):
        print(f"Running Model Part >>> {idx}")
        subModInput: dict[str] = {}
        for inputName in subMod.input:
            subModInput[inputName] = producedOutputs[inputName]

        subModOut: dict[str] = subMod(subModInput)
        for outName in subModOut:
            producedOutputs[outName] = subModOut[outName]

    diffNorm = tf.norm(producedOutputs["box_0"] - wholeModelOutput["box_0"])
    print(f"Yolo Model Test - Norm of Difference >> {diffNorm}")


if __name__ == "__main__":
    setUpTest()
    test_yolo()
    # testSavedModel()
