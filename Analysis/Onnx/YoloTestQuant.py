import os
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.quantize import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO
from YoloPPP import YoloPPP

CALIBRATION_DATA_PATH = "./coco128/images/train2017/"


class MyDataReader:
    def __init__(self, calibration_data_path: str):
        self.calib_data_path = calibration_data_path

        self.files = [
            entry.name for entry in os.scandir(calibration_data_path) if entry.is_file()
        ]
        self.max_files = 1
        self.files.sort()
        self.yolo_segmentation_ppp = YoloPPP(640, 640)
        self.idx = 0

    def get_next(self):
        if self.idx == self.max_files:
            print("Finished Calibration")
            return None
        print("Getting Image >> ", self.idx)

        image_path = os.path.join(
            self.calib_data_path, self.files[self.idx % self.max_files]
        )
        prep_dict = self.yolo_segmentation_ppp.preprocess(cv2.imread(image_path))
        prep_image = prep_dict["preprocessed_image"]

        self.idx += 1
        return {"images": prep_image}


def run_model(model_path: str, test_files: list[str], yolo_ppp: YoloPPP):

    time_array = np.zeros(len(test_files))
    out_array = np.zeros(len(test_files))

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.optimized_model_filepath = model_path.replace(".onnx", "_opt.onnx")
    sess = ort.InferenceSession(model_path)
    for idx, image_path in enumerate(test_files):
        prep_dict = yolo_ppp.preprocess(cv2.imread(image_path))
        prep_image = prep_dict["preprocessed_image"]

        start = time.perf_counter_ns()
        out = sess.run(None, {"images": prep_image})
        end = time.perf_counter_ns()
        time_array[idx] = (end - start) / 1e6
        # out_array[idx] = out[0]

    return time_array, out_array


def main():

    model = YOLO("yolo11n-seg.pt")
    model.export(format="onnx")

    model = onnx.load_model("./models/yolo11n-seg.onnx")
    node_name = []
    for node in model.graph.node:
        if node.op_type == "Concat":
            node_name.append(node.name)

    quant_pre_process(
        "./models/yolo11n-seg.onnx",
        output_model_path="./models/yolo11n-seg_pre_quant.onnx",
    )

    quantize_static(
        model_input="./models/yolo11n-seg_pre_quant.onnx",
        model_output="./models/yolo11n-seg_quant_QOp.onnx",
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibration_data_reader=MyDataReader(CALIBRATION_DATA_PATH),
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=node_name,
    )

    return
    # ["/model.23/Concat_6"]
    files = [
        os.path.join(CALIBRATION_DATA_PATH, entry.name)
        for entry in os.scandir(CALIBRATION_DATA_PATH)
        if entry.is_file()
    ]
    files.sort()
    test_files = files
    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640)
    time_array_quant, out_array = run_model(
        "./models/yolo11x-seg_quant.onnx", test_files, yolo_segmentation_ppp
    )
    print("Quant Time >> ", time_array_quant.mean())

    time_array_not_quant, out_array = run_model(
        "./models/yolo11x-seg.onnx", test_files, yolo_segmentation_ppp
    )
    print("Not Quant Time >> ", time_array_not_quant.mean())


if __name__ == "__main__":
    main()
