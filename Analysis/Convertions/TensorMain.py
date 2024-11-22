import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("yolov8n.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("yolov8n.pb")
