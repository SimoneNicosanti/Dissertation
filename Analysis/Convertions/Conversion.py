import keras_cv
import ModelParse

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xl_backbone_coco"  # We will use yolov8 small backbone with coco weights
)
print(len(backbone.layers))
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=10,
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)
models = ModelParse.modelParse(backbone, maxLayerNum=60)
for i, mod in enumerate(models):
    print(mod.summary())
    mod.save(f"./models/yolo_{i}.keras")
