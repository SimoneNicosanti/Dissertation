import csv

import keras
import keras_cv
from Flops.FlopsComputer import FlopsComputer
from Manipulation import Unnester
from Manipulation.NodeWrapper import NodeKey


def setUpClass():
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=backbone,
        fpn_depth=2,
    )

    unnestedModel: keras.Model = Unnester.unnestModel(yolo)
    unnestedModel.save("./models/UnnestedYolo.keras")


def test_yolo():
    model: keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")

    computer = FlopsComputer(model)
    flopsPerOp: dict[NodeKey, float, float, float] = computer.computeFloatOpsPerOp(
        [(1, 64, 64, 3)]
    )

    _, timesPerOp = computer.computeRunningTimes([(1, 64, 64, 3)], 25)
    outShapes = computer.computeOutputShapes([(1, 64, 64, 3)])

    with open("./other/Yolo_FLOPS.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["NodeKey", "FloatOps", "AvgTime", "OutShapes"])

        for opKey in flopsPerOp.keys():
            writer.writerow(
                [opKey, flopsPerOp[opKey], timesPerOp[opKey]] + outShapes[opKey]
            )


def test_deep_lab():
    backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
        preset="resnet50_v2_imagenet",
        input_shape=(224, 224) + (3,),
        load_weights=True,
    )
    segmenter = keras_cv.models.segmentation.DeepLabV3Plus(
        num_classes=20,
        backbone=backbone,
    )

    segmenter: keras.Model = Unnester.unnestModel(segmenter)
    segmenter.save("./models/UnnestedDeepLab.keras")

    segmenter = keras.saving.load_model("./models/UnnestedDeepLab.keras")

    computer = FlopsComputer(segmenter)
    flopsPerOp: dict[NodeKey, float, float, float] = computer.computeFloatOpsPerOp(
        [(1, 224, 224, 3)]
    )
    flopsPerModel: float = computer.computeFloatOpsPerModel([(1, 224, 224, 3)])
    print(sum(flopsPerOp.values()))
    print(flopsPerModel)


if __name__ == "__main__":
    setUpClass()

    test_yolo()
    # test_deep_lab()
