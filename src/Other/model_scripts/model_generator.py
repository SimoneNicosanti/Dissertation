import os
import shutil

import onnx
import onnx.tools
import onnx.tools.update_model_dims
import torch
from onnx.tools.update_model_dims import update_inputs_outputs_dims
from ultralytics import YOLO


def download_and_export(model_names: list[str]):
    for mod_name in model_names:
        model = YOLO(mod_name)
        # model.export(format="onnx", imgsz=640, dynamic=True)
        mod_name = mod_name.replace(".pt", "")
        reshaped_model = torch.onnx.export(
            model.model,
            input_names=["images"],
            output_names=["output0"],
            args=torch.zeros(1, 3, 640, 640),
            f="./" + mod_name + ".onnx",
            dynamic_axes={
                "images": {0: "batch"},
                "output0": {0: "batch"},
            },
        )

        # onnx.save_model(reshaped_model, "./" + mod_name + "_resh.onnx")

        # shutil.move("./" + mod_name + ".onnx", "../models/" + mod_name + ".onnx")
        # os.remove("./" + mod_name + ".pt")


def main():
    seg_models = [
        "yolo11n-seg.pt",
        # "yolo11s-seg.pt",
        # # "yolo11m-seg.pt",
        # "yolo11l-seg.pt",
        # "yolo11x-seg.pt",
    ]
    det_models = ["yolo11n.pt", "yolo11s.pt", "yolo11l.pt", "yolo11x.pt"]

    cls_models = [
        "yolo11n-cls.pt",
        # "yolo11s-cls.pt",
        # "yolo11l-cls.pt",
        # "yolo11x-cls.pt",
    ]
    download_and_export(cls_models)

    # download_and_export(seg_models)
    # download_and_export(det_models)


if __name__ == "__main__":
    main()
