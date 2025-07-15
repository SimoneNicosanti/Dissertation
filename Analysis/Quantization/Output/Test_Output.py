import os

import cv2
import numpy as np
import onnxruntime
import supervision as sv
import YoloPPP
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static

QUANTIZABLE_LAYERS = [
    "/model.3/conv/Conv",
    "/model.23/cv2.0/cv2.0.0/conv/Conv",
    "/model.23/cv2.0/cv2.0.1/conv/Conv",
    "/model.5/conv/Conv",
    "/model.1/conv/Conv",
    "/model.23/cv2.1/cv2.1.0/conv/Conv",
    "/model.7/conv/Conv",
    "/model.16/cv1/conv/Conv",
    "/model.2/cv2/conv/Conv",
    "/model.4/cv2/conv/Conv",
]

COCO_FILE_PATH = "../../../coco128/preprocessed"


class MyDataReader(CalibrationDataReader):

    def __init__(self, calibration_data: list[np.ndarray]):
        self.idx = 0
        self.tot_data = len(calibration_data)
        self.calibration_data = calibration_data

    def get_next(self):
        if self.idx == self.tot_data:
            return None
        input = self.calibration_data[self.idx]
        self.idx += 1
        return {"images": input}


def read_all_images():

    files = os.listdir(COCO_FILE_PATH)

    # Elenca solo i file (non le directory)
    files = [f for f in files if os.path.isfile(os.path.join(COCO_FILE_PATH, f))]
    images = []
    for file_name in files:
        file_path = os.path.join(COCO_FILE_PATH, file_name)
        image = np.load(file_path)["arr_0"]

        images.append(image)

    return images


def main():

    data_reader = MyDataReader(read_all_images())

    # quant_pre_process("./yolo11n.onnx", output_model_path="./yolo11n_pre_quant.onnx")

    quantize_static(
        model_input="./yolo11n_pre_quant.onnx",
        model_output="./yolo11n_quant.onnx",
        calibration_data_reader=data_reader,
        nodes_to_quantize=QUANTIZABLE_LAYERS,
    )

    yolo_ppp = YoloPPP.YoloPPP(640, 640)
    orig_image = cv2.imread("./Test_Image.jpg")
    pre_image = yolo_ppp.preprocess(orig_image)

    sess = onnxruntime.InferenceSession("./yolo11n_quant.onnx")
    quant_res = sess.run(None, input_feed={"images": pre_image})
    quant_out = quant_res[0]

    norm_res = onnxruntime.InferenceSession("./yolo11n.onnx").run(
        None, input_feed={"images": pre_image}
    )
    norm_out = norm_res[0]

    print(
        "Inf Norm >> ",
        np.linalg.norm(quant_out.ravel() - norm_out.ravel(), ord=np.inf),
    )
    print(
        "Inf Norm Elem Coord >> ",
        np.unravel_index(
            np.argmax(np.abs(quant_out - norm_out)), shape=quant_out.shape
        ),
    )
    print("Mean Diff >> ", np.mean(np.abs(quant_out - norm_out)))

    bboxes_1, _, _ = yolo_ppp.postprocess(
        orig_image,
        predictions=quant_out,
        prototypes=None,
        score_thr=0.5,
        iou_thr=0.5,
        num_classes=80,
    )

    bboxes_2, _, _ = yolo_ppp.postprocess(
        orig_image,
        predictions=norm_out,
        prototypes=None,
        score_thr=0.5,
        iou_thr=0.5,
        num_classes=80,
    )

    print(
        "Inf Norm >> ",
        np.linalg.norm(bboxes_1.ravel() - bboxes_2.ravel(), ord=np.inf),
    )
    print(
        "Inf Norm Elem Coord >> ",
        np.unravel_index(np.argmax(np.abs(bboxes_1 - bboxes_2)), shape=bboxes_1.shape),
    )
    print("Mean Diff >> ", np.mean(np.abs(bboxes_1 - bboxes_2)))

    if bboxes_1 is not None:
        detections = sv.Detections(
            xyxy=bboxes_1[:, :4],
            mask=None,
            confidence=bboxes_1[:, 4],
            class_id=bboxes_1[:, 5].astype(int),
        )

        mask_annotator = sv.MaskAnnotator()

        # Applica le annotazioni a un'immagine
        annotated_image = mask_annotator.annotate(
            scene=orig_image,  # la tua immagine (array NumPy)
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
            scene=orig_image,  # la tua immagine (array NumPy)
            detections=detections,  # le detections da disegnare
        )

        label_annotator = sv.LabelAnnotator()

        # Applica le annotazioni a un'immagine
        annotated_image = label_annotator.annotate(
            scene=orig_image,  # la tua immagine (array NumPy)
            detections=detections,  # le detections da disegnare
            labels=labels,
        )

        cv2.imwrite("./Test_Image_Out_Quant.jpg", annotated_image)

    pass


if __name__ == "__main__":
    main()
