import keras
import keras.src
import keras.src.ops.function
import keras_cv
import tensorflow as tf

backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone_coco")
backbone.name = "backbone"

images = tf.ones(shape=(1, 512, 512, 3))
labels = {
    "boxes": tf.constant(
        [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        dtype=tf.float32,
    ),
    "classes": tf.constant([[1, 1, 1]], dtype=tf.int64),
}

model = keras_cv.models.YOLOV8Detector(
    num_classes=20,
    bounding_box_format="xywh",
    backbone=backbone,
    fpn_depth=2,
)

print(model.get_layer("functional").get_layer(index=1).input)
print(model.get_layer("functional").output)
print(model.get_layer(index=2).input)
