import os
import shutil

from ultralytics import YOLO


def download_and_export(model_names: list[str]):
    for mod_name in model_names:
        model = YOLO(mod_name)
        model.export(format="onnx", nms=True)
        mod_name = mod_name.replace(".pt", "")
        shutil.move("./" + mod_name + ".onnx", "../models_1/" + mod_name + ".onnx")
        os.remove("./" + mod_name + ".pt")


def main():
    seg_models = [
        "yolo11n-seg.pt",
        # "yolo11s-seg.pt",
        # # "yolo11m-seg.pt",
        # "yolo11l-seg.pt",
        # "yolo11x-seg.pt",
    ]
    det_models = ["yolo11n.pt", "yolo11s.pt", "yolo11l.pt", "yolo11x.pt"]

    download_and_export(seg_models)
    download_and_export(det_models)


if __name__ == "__main__":
    main()
