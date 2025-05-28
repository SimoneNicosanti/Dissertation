import time

import grpc

from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileStub

execution_profile_request = ExecutionProfileRequest(
    model_id=ModelId(model_name="yolo11n-seg")
)

port = ConfigReader.ConfigReader("../../../config/config.ini").read_int(
    "ports", "EXECUTION_PROFILER_PORT"
)
profiler_stub = ExecutionProfileStub(grpc.insecure_channel(f"localhost:{port}"))

start = time.perf_counter_ns()
profiler_response: ExecutionProfileResponse = profiler_stub.profile_execution(
    execution_profile_request
)
end = time.perf_counter_ns()

print("Profile Time >> ", (end - start) * 1e-9)
