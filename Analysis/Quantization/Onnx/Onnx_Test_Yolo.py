import onnx
import onnxruntime as rt
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

yolo_model = YOLO("yolo11n.pt")
yolo_model.export(format=".onnx")

onnx_model = onnx.load("yolo11n.onnx")

sess_options = rt.SessionOptions()
# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
# sess_options.optimized_model_filepath = "./yolo11n_opt.onnx"

# quant_pre_process("yolo11n.onnx", output_model_path="yolo11n_pre_quant.onnx")

# session = rt.InferenceSession("./yolo11n.onnx", sess_options)
