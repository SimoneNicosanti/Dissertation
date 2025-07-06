import os
import shutil

import numpy as np
import onnx
import onnx.checker
import onnxruntime.tools.make_dynamic_shape_fixed
from ultralytics import YOLO


def download_and_export(model_names: list[str]):
    for mod_name in model_names:
        model = YOLO(mod_name)
        model.export(format="onnx", dynamic=True)

        mod_name = mod_name.replace(".pt", "")
        # shutil.move("./" + mod_name + ".onnx", "../models/" + mod_name + ".onnx")
        # os.remove("./" + mod_name + ".pt")


def main():
    download_and_export(["yolo11x-seg.pt"])
    model = onnx.load_model("yolo11x-seg.onnx")
    onnxruntime.tools.make_dynamic_shape_fixed.make_input_shape_fixed(
        model.graph, input_name="images", fixed_shape=[-1, 3, 640, 640]
    )
    onnx.save_model(model, "yolo11x-seg_fixed.onnx")

    # fixed_model = onnx.load_model("yolo11n-seg_fixed.onnx")

    # onnx.checker.check_model(fixed_model, full_check=True)

    # sess = onnxruntime.InferenceSession("yolo11n-seg_fixed.onnx")
    # out = sess.run(
    #     None, input_feed={"images": np.zeros((10, 3, 640, 640), dtype=np.float32)}
    # )

    # print(out[0].shape)
    pass


if __name__ == "__main__":
    main()
