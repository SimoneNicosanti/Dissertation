import grpc

from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId, OptimizedPlan
from proto_compiled.deployment_pb2 import DeploymentRequest
from proto_compiled.deployment_pb2_grpc import DeploymentServicer
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub
from proto_compiled.register_pb2 import AllServerInfo, ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileStub


class DeploymentServer(DeploymentServicer):

    def __init__(self):
        super().__init__()

        model_profiler_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "MODEL_PROFILER_ADDR"
        )
        model_profiler_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "MODEL_PROFILER_PORT"
        )
        self.model_profiler_chann = grpc.insecure_channel(
            "{}:{}".format(model_profiler_addr, model_profiler_port)
        )

        registry_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "REGISTRY_ADDR"
        )
        registry_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "REGISTRY_PORT"
        )
        self.registry_chann = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )

    def do_models_profiling(
        self, models_ids: list[ModelId]
    ) -> dict[ModelId, ProfileResponse]:

        model_profiles: dict[ModelId, ProfileResponse] = {}
        model_profiler: ModelProfileStub = ModelProfileStub(self.model_profiler_chann)
        for model_id in models_ids:
            model_profile_req: ProfileRequest = ProfileRequest(model_id=model_id)
            model_profile_res: ProfileResponse = model_profiler.profile_model(
                model_profile_req
            )

            model_profiles[model_id] = model_profile_res

        return model_profiles

    def do_servers_profiling(
        self, models_ids: list[ModelId]
    ) -> dict[ServerId, dict[ModelId, ExecutionProfileResponse]]:

        registry: RegisterStub = RegisterStub(self.registry_chann)
        all_server_info: AllServerInfo = registry.get_all_servers_info()

        server_profiles: dict[ServerId, dict[ModelId, ExecutionProfileResponse]] = {}
        for server_info in all_server_info.all_server_info:
            server_addr = server_info.reachability_info.ip_address
            server_profile_port = ConfigReader.ConfigReader(
                "./config/config.ini"
            ).read_int("ports", "EXECUTION_PROFILER_PORT")
            server_id: str = server_info.server_id.server_id

            server_conn = grpc.insecure_channel(
                "{}:{}".format(server_addr, server_profile_port)
            )
            server_profiles[server_id] = {}

            server_profiler: ExecutionProfileStub = ExecutionProfileStub(server_conn)
            for model_id in models_ids:

                execution_profile_response: ExecutionProfileRequest = (
                    server_profiler.profile_execution(
                        ExecutionProfileRequest(model_id=model_id)
                    )
                )

                server_profiles[server_id][model_id] = execution_profile_response
            server_conn.close()

        return server_profiles

    def deploy_model(self, deployment_req: DeploymentRequest, context):

        models_ids: list[ModelId] = deployment_req.models_ids

        ## Model and Server profiling will be done only once
        ## Then they will be saved in server and model file systems
        model_profiles: dict[ModelId, ProfileResponse] = self.do_models_profiling(
            models_ids
        )
        server_profiles: dict[ServerId, dict[ModelId, ExecutionProfileResponse]] = (
            self.do_servers_profiling(models_ids)
        )

        ## TODO Implement Bandwidth Stats Get
        ## TODO Implement Optimization Call
        ## TODO Implement Model Division Call
        ## TODO Return Optimized Plan

        return super().deploy_model(deployment_req, context)

    def deploy_plan(self, optimized_plan: OptimizedPlan, context):
        ## TODO Call Fetchers to get their sub models

        return super().deploy_plan(optimized_plan, context)
