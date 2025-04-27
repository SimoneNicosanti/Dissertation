import numpy as np
import onnx
import onnx_tool

model = onnx.load_model("yolo11n_nms.onnx")
m = onnx_tool.Model(m=model)

input_dict = {"images": np.zeros(shape=(1, 3, 640, 640), dtype=np.float32)}
m.graph.shape_infer(input_dict)
m.graph.profile()

m.graph.print_node_map(metric="FLOPs")
