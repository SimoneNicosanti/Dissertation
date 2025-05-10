import time

import onnx
from Profiler import ExecutionProfiler

onnx_model = onnx.load_model(
    "../../Other/models/yolo11x-seg.onnx", load_external_data=False
)

print(len(onnx_model.SerializeToString()))

print(len(onnx_model.graph.node))

exec_profiler = ExecutionProfiler()
start = time.perf_counter_ns()
exec_dict = exec_profiler.profile_per_layer_exec_time(onnx_model, 5)
end = time.perf_counter_ns()

print("Profile Time >> ", (end - start) * 1e-9)

for layer, exec_time in exec_dict.items():
    print(layer, exec_time)
