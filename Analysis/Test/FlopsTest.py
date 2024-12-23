import csv

import keras
import keras_cv
from Flops import ComputeFlops
from Manipulation import Unnester


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

    # ComputeFlops.computeRunningTimes(model, (1, 64, 64, 3), 10)
    # ComputeFlops.computeFloatOperationsPerOperation(model, (1, 64, 64, 3))

    flopsPerOp: dict[str, tuple[float, float, float]] = ComputeFlops.computeFlopsPerOp(
        model, (1, 64, 64, 3), 25
    )

    with open("./other/Yolo_FLOPS.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["OpName", "FloatOps", "AvgTime", "FLOPS"])

        for opName in flopsPerOp:
            writer.writerow([opName, *flopsPerOp[opName]])


def test_mobile_net():
    inpSize: int = 64
    model: keras.Model = keras.applications.MobileNetV3Large()
    flopsPerOp: dict[str, tuple[float, float, float]] = ComputeFlops.computeFlopsPerOp(
        model, (1, inpSize, inpSize, 3), 25
    )

    with open("./other/MobileNetV3_Large_FLOPS.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["OpName", "InpSize", "FloatOps", "AvgTime", "FLOPS"])

        for opName in flopsPerOp:
            writer.writerow([opName, inpSize, *flopsPerOp[opName]])


if __name__ == "__main__":
    setUpClass()
    
    test_yolo()
    test_mobile_net()
