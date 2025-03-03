import cv2
import onnxruntime as ort
from ultralytics import YOLO
from YoloDetectionPPP import YoloDetectionPPP
from YoloSegmentationPPP import YoloSegmentationPPP


def run_original(model_name, input_name):
    original_model = YOLO(model_name)
    original_model(input_name, save=True)
    original_model.export(format="onnx", simplify=True)


def test_detection():
    # run_original("yolo11n.pt", "./bus.jpg")

    yolo_v8_ppp = YoloDetectionPPP(640, 640)
    original_image = cv2.imread("./bus.jpg")
    pre_image = yolo_v8_ppp.preprocess(original_image)["preprocessed_image"]

    sess = ort.InferenceSession("./models/yolo11n.onnx")
    model_out = sess.run(None, input_feed={"images": pre_image})

    post_image = yolo_v8_ppp.postprocess(original_image, model_out, 0.5, 0.5)

    cv2.imwrite("./results/ppp/result_det.jpg", post_image)


def test_segmentation():
    # run_original("yolo11n-seg.pt", "./bus.jpg")

    orig_image = cv2.imread("./bus.jpg")

    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640)

    # Pre-process
    preprocess_dict = yolo_segmentation_ppp.preprocess(orig_image)
    pre_image = preprocess_dict["preprocessed_image"]

    # # Ort inference
    model_output = ort.InferenceSession("./models/yolo11n-seg.onnx").run(
        None, {"images": pre_image}
    )

    post_image = yolo_segmentation_ppp.postprocess(
        orig_image,
        model_output,
        0.5,
        0.5,
        ratio=preprocess_dict["ratio"],
        pad_w=preprocess_dict["pad_w"],
        pad_h=preprocess_dict["pad_h"],
        nm=32,
    )

    cv2.imwrite("./results/ppp/result_seg.jpg", post_image)


if __name__ == "__main__":

    test_detection()
    test_segmentation()
