import json
import os
from typing import Iterator

import grpc
import onnx

from Common import ConfigReader
from proto_compiled.common_pb2 import ComponentId, ModelId
from proto_compiled.model_pool_pb2 import LayerPullResponse, PullRequest
from proto_compiled.model_pool_pb2_grpc import ModelPoolStub
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileServicer
from Server.Profiler.Profiler import ExecutionProfiler


class ExecutionProfileServer(ExecutionProfileServicer):
    def __init__(self):

        model_pool_addr = ConfigReader.ConfigReader().read_str(
            "addresses", "MODEL_POOL_ADDR"
        )
        model_pool_port = ConfigReader.ConfigReader().read_int(
            "ports", "MODEL_POOL_PORT"
        )
        self.model_pool_chann = grpc.insecure_channel(
            "{}:{}".format(model_pool_addr, model_pool_port)
        )

        self.model_profiles_dir = ConfigReader.ConfigReader().read_str(
            "server_dirs", "PROFILES_DIR"
        )

        self.layer_run_times = ConfigReader.ConfigReader().read_int(
            "server_profiler", "LAYER_RUN_TIMES"
        )

        pass

    def profile_execution(
        self, request: ExecutionProfileRequest, context
    ) -> ExecutionProfileResponse:
        print("Received Execution Profile Request")
        print(request.model_id)
        model_profile = self.read_profile(request.model_id)

        if model_profile is not None:
            return ExecutionProfileResponse(profile=model_profile)
        else:
            model_profile = {}

        profiler = ExecutionProfiler()

        model_id: ModelId = request.model_id

        for layer_name, layer_model in self.retrieve_layers(model_id):
            layer_exec_times = profiler.profile_exec_time(
                layer_model, self.layer_run_times
            )

            print(f"Profiled {layer_name} >> {layer_exec_times}")
            model_profile[layer_name] = layer_exec_times

        self.save_profile(model_id, model_profile)

        return ExecutionProfileResponse(profile=str(model_profile))

    def retrieve_layers(self, model_id: ModelId) -> Iterator[onnx.ModelProto]:
        model_pool_stub = ModelPoolStub(self.model_pool_chann)
        component_id = ComponentId(model_id=model_id, server_id="", component_idx="")

        pull_request = PullRequest(component_id=component_id)

        response_stream: Iterator[LayerPullResponse] = (
            model_pool_stub.pull_layer_models(pull_request)
        )

        layer_name = None
        layer_model_bytes = None
        for layer_pull_response in response_stream:
            curr_layer_name = layer_pull_response.layer_name

            if layer_name != curr_layer_name:

                if layer_name is not None and layer_model_bytes is not None:
                    layer_model: onnx.ModelProto = onnx.load_from_string(
                        bytes(layer_model_bytes)
                    )
                    yield layer_name, layer_model

                layer_name = curr_layer_name
                layer_model_bytes = bytearray()

            layer_model_bytes.extend(layer_pull_response.model_chunk.chunk_data)

        if layer_name is not None and layer_model_bytes is not None:
            layer_model: onnx.ModelProto = onnx.load_from_string(
                bytes(layer_model_bytes)
            )
            yield layer_name, layer_model

    def read_profile(self, model_id: ModelId) -> str:

        file_name = self.build_file_name(model_id)
        file_path = os.path.join(self.model_profiles_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return file.read()

        return None

    def save_profile(self, model_id: ModelId, profile: dict):
        file_name = self.build_file_name(model_id)
        file_path = os.path.join(self.model_profiles_dir, file_name)

        profile_str = json.dumps(profile)

        with open(file_path, "w") as file:
            file.write(profile_str)

    def build_file_name(self, model_id: ModelId) -> str:
        return f"{model_id.model_name}_depl_{model_id.deployer_id}_profile.json"
