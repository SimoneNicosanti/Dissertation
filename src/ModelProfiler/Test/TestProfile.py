import json

import grpc
import networkx as nx

from CommonProfile import ProfileCoder
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub

model_name = "yolo11n-seg"

model_profiler = ModelProfileStub(grpc.insecure_channel("localhost:50004"))


profile_request = ProfileRequest(
    model_id=ModelId(model_name=model_name, deployer_id="0")
)
profile_response: ProfileResponse = model_profiler.profile_model(profile_request)

profile = nx.node_link_graph(json.loads(profile_response.model_profile))

decoded_profile = ProfileCoder.decode_model_profile(profile)
