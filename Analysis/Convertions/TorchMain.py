import onnx
import torch
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.export(format="onnx")
    pass


if __name__ == "__main__":
    main()
