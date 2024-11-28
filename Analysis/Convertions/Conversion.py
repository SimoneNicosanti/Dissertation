import keras_cv
import ModelParse

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=10,
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)
print(yolo.summary())
models = ModelParse.modelParse(yolo, maxLayerNum=25)
for i, mod in enumerate(models):
    mod.save(f"./models/yolo_{i}.keras")
