import os

import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader
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

    # quant_pre_process("./yolo11n.onnx", output_model_path="./yolo11n_pre_quant.onnx")

    for i in range(0, 9):

        data_reader = MyDataReader(read_all_images())

        quan_sub_list = np.random.choice(
            QUANTIZABLE_LAYERS, size=i + 1, replace=False
        ).tolist()
        print(quan_sub_list)

        quantize_static(
            model_input="./yolo11n_pre_quant.onnx",
            model_output=f"./yolo11n_quant_{i}.onnx",
            calibration_data_reader=data_reader,
            nodes_to_quantize=quan_sub_list,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )


if __name__ == "__main__":
    main()
