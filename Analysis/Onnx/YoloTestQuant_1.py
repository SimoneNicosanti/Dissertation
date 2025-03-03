import os
import time

import numpy as np
import onnxruntime as ort


def run_model(model_path: str, test_num: int):

    time_array = np.zeros(test_num)
    out_array = np.zeros(test_num)

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.optimized_model_filepath = model_path.replace(".onnx", "_opt.onnx")
    sess = ort.InferenceSession(model_path)
    for idx in range(test_num):
        prep_image = np.zeros(shape=(1, 3, 640, 640), dtype=np.float32)

        start = time.perf_counter_ns()
        sess.run(None, {"images": prep_image})
        end = time.perf_counter_ns()
        time_array[idx] = (end - start) / 1e6
        # out_array[idx] = out[0]

    return time_array, out_array


def main():
    # model = onnx.load_model("./models/yolo11n-seg.onnx")
    # node_name = []
    # for node in model.graph.node:
    #     if node.op_type == "Concat":
    #         node_name.append(node.name)

    # quant_pre_process(
    #     "./models/yolo11n-seg.onnx",
    #     output_model_path="./models/yolo11n-seg_pre_quant.onnx",
    # )

    # quantize_static(
    #     model_input="./models/yolo11n-seg_pre_quant.onnx",
    #     model_output="./models/yolo11n-seg_quant.onnx",
    #     quant_format=QuantFormat.QOperator,
    #     activation_type=QuantType.QUInt8,
    #     weight_type=QuantType.QUInt8,
    #     calibration_data_reader=MyDataReader(CALIBRATION_DATA_PATH),
    #     calibrate_method=CalibrationMethod.MinMax,
    #     nodes_to_exclude=node_name,
    # )
    # # ["/model.23/Concat_6"]
    # files = [
    #     os.path.join(CALIBRATION_DATA_PATH, entry.name)
    #     for entry in os.scandir(CALIBRATION_DATA_PATH)
    #     if entry.is_file()
    # ]
    # files.sort()
    # test_files = files
    # yolo_segmentation_ppp = YoloSegmentationPPP(640, 640)
    time_array_quant, out_array = run_model("./models/yolo11x-seg_quant.onnx", 200)
    print("Quant Time >> ", time_array_quant.mean())

    # time_array_not_quant, out_array = run_model("./models/yolo11x-seg.onnx", 100)
    # print("Not Quant Time >> ", time_array_not_quant.mean())


if __name__ == "__main__":
    main()
