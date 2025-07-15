import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from ultralytics import YOLO
from YoloPPP import YoloPPP


def run_original(model_name, input_name):
    original_model = YOLO(model_name)
    original_model(input_name, save=True)
    original_model.export(format="onnx", simplify=True)


def test_detection():
    # run_original("yolo11n.pt", "./bus.jpg")

    yolo_v8_ppp = YoloPPP(640, 640)
    original_image = cv2.imread("./bus.jpg")
    pre_image = yolo_v8_ppp.preprocess(original_image)

    sess = ort.InferenceSession("yolo11n.onnx")
    model_out = sess.run(None, input_feed={"images": pre_image})

    bboxes, _, _ = yolo_v8_ppp.postprocess(
        original_image,
        predictions=model_out[0],
        prototypes=None,
        score_thr=0.5,
        iou_thr=0.5,
        num_classes=80,
    )

    detections = sv.Detections(
        xyxy=bboxes[:, :4], confidence=bboxes[:, 4], class_id=bboxes[:, 5].astype(int)
    )

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections.class_id, detections.confidence)
    ]

    box_annotator = sv.BoxAnnotator()

    # Applica le annotazioni a un'immagine
    annotated_image = box_annotator.annotate(
        scene=original_image,  # la tua immagine (array NumPy)
        detections=detections,  # le detections da disegnare
    )

    label_annotator = sv.LabelAnnotator()

    # Applica le annotazioni a un'immagine
    annotated_image = label_annotator.annotate(
        scene=original_image,  # la tua immagine (array NumPy)
        detections=detections,  # le detections da disegnare
        labels=labels,
    )

    cv2.imwrite("./results/ppp/result_det.jpg", annotated_image)

    # cv2.imwrite("./results/ppp/result_det.jpg", post_image)


def test_segmentation():
    # run_original("yolo11n-seg.pt", "./bus.jpg")

    # run_original("yolo11n.pt", "./bus.jpg")

    yolo_v8_ppp = YoloPPP(640, 640)
    original_image = cv2.imread("./bus.jpg")
    pre_image = yolo_v8_ppp.preprocess(original_image)

    sess = ort.InferenceSession("yolo11n-seg_fixed.onnx")
    model_out = sess.run(None, input_feed={"images": pre_image})

    bboxes, masks, _ = yolo_v8_ppp.postprocess(
        original_image,
        predictions=model_out[0],
        prototypes=model_out[1],
        score_thr=0.5,
        iou_thr=0.5,
        num_classes=80,
    )

    if bboxes is not None and masks is not None:

        detections = sv.Detections(
            xyxy=bboxes[:, :4],
            mask=masks,
            confidence=bboxes[:, 4],
            class_id=bboxes[:, 5].astype(int),
        )

        mask_annotator = sv.MaskAnnotator()

        # Applica le annotazioni a un'immagine
        annotated_image = mask_annotator.annotate(
            scene=original_image,  # la tua immagine (array NumPy)
            detections=detections,  # le detections da disegnare
        )

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(
                detections.class_id, detections.confidence
            )
        ]

        box_annotator = sv.BoxAnnotator()

        # Applica le annotazioni a un'immagine
        annotated_image = box_annotator.annotate(
            scene=original_image,  # la tua immagine (array NumPy)
            detections=detections,  # le detections da disegnare
        )

        label_annotator = sv.LabelAnnotator()

        # Applica le annotazioni a un'immagine
        annotated_image = label_annotator.annotate(
            scene=original_image,  # la tua immagine (array NumPy)
            detections=detections,  # le detections da disegnare
            labels=labels,
        )

        cv2.imwrite("./results/ppp/result_seg.jpg", annotated_image)

    else:
        cv2.imwrite("./results/ppp/result_seg.jpg", original_image)


if __name__ == "__main__":

    # test_detection()
    test_segmentation()
