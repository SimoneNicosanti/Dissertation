import json

import grpc

from Common import ConfigReader
from CommonProfile.ExecutionProfile import (
    ModelExecutionProfile,
    ServerExecutionProfile,
    ServerExecutionProfilePool,
)
from CommonProfile.NodeId import NodeId
from proto_compiled.common_pb2 import Empty, ModelId
from proto_compiled.register_pb2 import AllServerInfo
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileStub


class ExecutionProfileBuilder:
    def __init__(self):

        registry_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "REGISTRY_ADDR"
        )
        registry_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "REGISTRY_PORT"
        )
        self.registry_chann = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )

        pass

    def build_execution_profiles(
        self, models_ids: list[ModelId]
    ) -> ServerExecutionProfilePool:

        registry: RegisterStub = RegisterStub(self.registry_chann)
        all_server_info: AllServerInfo = registry.get_all_servers_info(Empty())

        server_exec_profile_pool = ServerExecutionProfilePool()

        for server_info in all_server_info.all_server_info:
            server_addr = server_info.reachability_info.ip_address
            server_profile_port = ConfigReader.ConfigReader(
                "./config/config.ini"
            ).read_int("ports", "EXECUTION_PROFILER_PORT")

            server_conn = grpc.insecure_channel(
                "{}:{}".format(server_addr, server_profile_port)
            )

            server_exec_profile = ServerExecutionProfile()

            server_profiler: ExecutionProfileStub = ExecutionProfileStub(server_conn)
            for model_id in models_ids:

                execution_profile_response: ExecutionProfileResponse = (
                    server_profiler.profile_execution(
                        ExecutionProfileRequest(model_id=model_id)
                    )
                )

                execution_profile_dict = json.loads(execution_profile_response.profile)

                server_exec_profile.put_model_execution_profile(
                    model_id.model_name,
                    ModelExecutionProfile().decode(execution_profile_dict),
                )
                print("Decoded")

            server_exec_profile_pool.put_execution_profiles_for_server(
                NodeId(server_info.server_id.server_id), server_exec_profile
            )

            server_conn.close()

        return server_exec_profile_pool

        pass
