from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelProfile import ModelProfile
from CommonProfile.NetworkProfile import NetworkProfile
from Deployer.Builders import ModelProfileBuilder
from Deployer.Builders.ExecutionProfileBuilder import ExecutionProfileBuilder
from Deployer.Builders.ModelProfileBuilder import ModelProfileBuilder
from Deployer.Builders.NetworkProfileBuilder import NetworkProfileBuilder
from Deployer.Optimization.OptimizerCaller import OptimizerCaller
from proto_compiled.common_pb2 import ModelId, OptimizedPlan
from proto_compiled.deployment_pb2 import DeploymentRequest, DeploymentResponse
from proto_compiled.deployment_pb2_grpc import DeploymentServicer


class DeploymentServer(DeploymentServicer):

    def __init__(self):
        super().__init__()

        self.model_profile_builder = ModelProfileBuilder()
        self.execution_profile_builder = ExecutionProfileBuilder()
        self.network_profile_builder = NetworkProfileBuilder()

        self.optimizer_caller = OptimizerCaller()

    def deploy_model(self, deployment_req: DeploymentRequest, context):
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

        self.optimizer_caller.call_optimizer(
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

        return DeploymentResponse(
            optimized_plan=OptimizedPlan(deployer_id="", plans_map={})
        )

    def deploy_plan(self, optimized_plan: OptimizedPlan, context):
        ## TODO Call Fetchers to get their sub models

        return super().deploy_plan(optimized_plan, context)
