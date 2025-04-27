import copy
import json

import networkx as nx
import onnx

from ModelManager.Profile import ProfileSaver
from ModelManager.Profile.OnnxModelProfiler import OnnxModelProfiler

profiler = OnnxModelProfiler("")

model = onnx.load("../models/yolo11n.onnx")
model_graph = nx.DiGraph()
profiler.init_model_graph(model_graph, model)

profiler.do_profile(model_graph, model)

model_graph.graph["name"] = "yolo11n"

to_save_graph = copy.deepcopy(model_graph)
label_mapping = {node_id: node_id.node_name for node_id in to_save_graph.nodes}
to_save_graph = nx.relabel_nodes(to_save_graph, label_mapping)
model_name = model_graph.graph["name"]

with open("yolo11n.json", "w") as json_file:
    json.dump(nx.node_link_data(to_save_graph), json_file)

ProfileSaver.save_profile(model_graph)
