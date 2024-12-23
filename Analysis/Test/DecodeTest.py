from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import decode_regression_to_boxes, get_anchors, dist2bbox
from keras_cv.src.bounding_box.converters import convert_format
from keras_cv.src.backend import ops
import keras_cv
import keras
import tensorflow as tf

def yolo_decoder_function(pred : dict, images) :
    boxes = pred["boxes"]
    scores = pred["classes"]

    boxes = decode_regression_to_boxes(boxes)

    anchor_points, stride_tensor = get_anchors(image_shape=images.shape[1:])
    stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

    box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
    box_preds = convert_format(
        box_preds,
        source="xyxy",
        target="xyxy",
        images=images,
    )

    return keras_cv.layers.NonMaxSuppression(
                bounding_box_format="xyxy",
                from_logits=False,
                confidence_threshold=0.2,
                iou_threshold=0.7,
            )(box_preds, scores)



def test_yolo_with_decode():
    images = tf.ones(shape=(1, 512, 512, 3))

    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")

    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=2,
    )

    out_1 = yolo(images)
    decoded = yolo_decoder_function(out_1, images)
    
    print(decoded)
    print(yolo.predict(images))


if __name__ == "__main__":
    test_yolo_with_decode()