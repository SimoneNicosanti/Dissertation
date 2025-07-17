import json
import os
import sys
from typing import Iterator

import grpc
import onnx

from Common import ConfigReader
from CommonIds.NodeId import NodeId
from CommonProfile.ExecutionProfile import ModelExecutionProfile
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
        print("\t ", request.model_id)
        model_exec_profile = self.read_profile(request.model_id)

        model_id: ModelId = request.model_id

        if model_exec_profile is None:
            print("\t Cache Miss")
            model_exec_profile = ModelExecutionProfile()

            profiler = ExecutionProfiler()

            tot_quant = 0
            tot_not_quant = 0
            for layer_name, layer_model, is_quantized in self.retrieve_layers(model_id):

                layer_exec_time = profiler.profile_exec_time(
                    layer_model, self.layer_run_times, is_quantized
                )
                print(
                    "\t\t Profiled {} with Quantization {} >> {}%.3f".format(
                        layer_name, is_quantized, layer_exec_time
                    )
                )

                node_id = NodeId(layer_name)
                model_exec_profile.put_layer_execution_profile(
                    node_id, layer_exec_time, is_quantized
                )

                if is_quantized:
                    tot_quant += layer_exec_time
                else:
                    tot_not_quant += layer_exec_time

            print(f"\t Total quantized time: {tot_quant}")
            print(f"\t Total not quantized time: {tot_not_quant}")

            # model_exec_profile.set_total_execution_time(tot_quant + tot_not_quant)

        print("\t Done Profiling for model {}".format(model_id.model_name))
        self.save_profile(model_id, model_exec_profile)

        model_exec_profile_json = json.dumps(model_exec_profile.encode())
        return ExecutionProfileResponse(profile=model_exec_profile_json)

    def retrieve_layers(
        self, model_id: ModelId
    ) -> Iterator[tuple[str, onnx.ModelProto, bool]]:
        model_pool_stub = ModelPoolStub(self.model_pool_chann)
        component_id = ComponentId(model_id=model_id, server_id="", component_idx="")

        pull_request = PullRequest(component_id=component_id)

        response_stream: Iterator[LayerPullResponse] = (
            model_pool_stub.pull_layer_models(pull_request)
        )

        layer_name = None
        layer_model_bytes = None
        is_quantized = None
        for layer_pull_response in response_stream:
            curr_layer_name = layer_pull_response.layer_name

            if layer_name != curr_layer_name:

                if layer_name is not None and layer_model_bytes is not None:
                    layer_model: onnx.ModelProto = onnx.load_from_string(
                        bytes(layer_model_bytes)
                    )
                    yield layer_name, layer_model, is_quantized

                layer_name = curr_layer_name
                layer_model_bytes = bytearray()
                is_quantized = layer_pull_response.is_quantized

            layer_model_bytes.extend(layer_pull_response.model_chunk.chunk_data)

        if layer_name is not None and layer_model_bytes is not None:
            layer_model: onnx.ModelProto = onnx.load_from_string(
                bytes(layer_model_bytes)
            )
            yield layer_name, layer_model, is_quantized

    def read_profile(self, model_id: ModelId) -> ModelExecutionProfile:
        file_name = self.build_file_name(model_id)
        file_path = os.path.join(self.model_profiles_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                encoded_execution_profile = json.load(json_file)
                model_execution_profile = ModelExecutionProfile().decode(
                    encoded_execution_profile
                )
                return model_execution_profile

        return None

    def save_profile(
        self,
        model_id: ModelId,
        model_execution_profile: ModelExecutionProfile,
    ):
        file_name = self.build_file_name(model_id)
        file_path = os.path.join(self.model_profiles_dir, file_name)
        encoded_model_execution_profile = model_execution_profile.encode()
        with open(file_path, "w") as json_file:
            json.dump(encoded_model_execution_profile, json_file)

    def build_file_name(self, model_id: ModelId) -> str:
        return f"{model_id.model_name}_profile.json"
