import json

from CommonPlan.WholePlan import WholePlan
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelProfile import ModelProfile
from CommonProfile.NetworkProfile import NetworkProfile
from Deployer.Builders.ExecutionProfileBuilder import ExecutionProfileBuilder
from Deployer.Builders.ModelProfileBuilder import ModelProfileBuilder
from Deployer.Builders.NetworkProfileBuilder import NetworkProfileBuilder
from Deployer.Caller.DividerCaller import ModelDivider
from Deployer.Caller.OptimizerCaller import OptimizerCaller
from proto_compiled.common_pb2 import ModelId, OptimizedPlan
from proto_compiled.deployment_pb2 import (
    DeploymentRequest,
    DeploymentResponse,
    ProducePlanRequest,
    ProducePlanResponse,
)
from proto_compiled.deployment_pb2_grpc import DeploymentServicer


class DeploymentServer(DeploymentServicer):

    def __init__(self):
        super().__init__()

        self.model_profile_builder = ModelProfileBuilder()
        self.execution_profile_builder = ExecutionProfileBuilder()
        self.network_profile_builder = NetworkProfileBuilder()

        self.optimizer_caller = OptimizerCaller()
        self.divider_caller = ModelDivider()
        # self.fetcher_caller = FetcherCaller()

    def produce_plan(self, deployment_req: ProducePlanRequest, context):
        print("Received Deployment Request")
        models_ids: list[ModelId] = deployment_req.models_ids

        ## Model and Server profiling will be done only once
        ## Then they will be saved in server and model file systems
        ## In case a deployer local cache can be implemented in order to avoid other requests
        models_profiles: list[ModelProfile] = (
            self.model_profile_builder.build_model_profiles(models_ids)
        )
        print("Built Models Profiles")

        server_exec_profiles_pool: ServerExecutionProfilePool = (
            self.execution_profile_builder.build_execution_profiles(models_ids)
        )
        print("Built Server Profiles")

        network_profile: NetworkProfile = (
            self.network_profile_builder.build_network_profile()
        )
        print("Built Network Profile")

        whole_plan: WholePlan = self.optimizer_caller.call_optimizer(
            models_profiles,
            network_profile,
            server_exec_profiles_pool,
            deployment_req.latency_weight,
            deployment_req.energy_weight,
            deployment_req.device_max_energy,
            deployment_req.requests_number,
            deployment_req.max_noises,
            deployment_req.start_server,
        )

        return ProducePlanResponse(optimized_plan=json.dumps(whole_plan.encode()))

    def deploy_plan(self, deployment_request: DeploymentRequest, context):
        print("Received Plan Deployment Request")
        whole_plan: WholePlan = WholePlan.decode(
            json.loads(deployment_request.optimized_plan)
        )

        self.divider_caller.divide_model(whole_plan)
        # self.fetcher_caller.distribute_plan(whole_plan)

        ## TODO Model Division
        ## TODO Fetcher Calls

        return DeploymentResponse()
