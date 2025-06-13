import json
import time

import grpc
import networkx as nx

from CommonProfile.ModelProfile import ModelProfile
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub

model_name = "yolo11s-seg"

model_profiler = ModelProfileStub(grpc.insecure_channel("localhost:50004"))

profile_request = ProfileRequest(model_id=ModelId(model_name=model_name))

start = time.perf_counter_ns()
profile_response: ProfileResponse = model_profiler.profile_model(profile_request)
end = time.perf_counter_ns()

print("Profile Time >> ", (end - start) * 1e-9)

transformed_dict = json.loads(profile_response.model_profile)

model_profile = ModelProfile.decode(transformed_dict)
