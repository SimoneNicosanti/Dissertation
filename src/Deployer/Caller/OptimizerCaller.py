import json

import grpc

from Common import ConfigReader
from CommonPlan.WholePlan import WholePlan
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelProfile import ModelProfile
from CommonProfile.NetworkProfile import NetworkProfile
from proto_compiled.optimizer_pb2 import OptimizationRequest, OptimizationResponse
from proto_compiled.optimizer_pb2_grpc import OptimizationStub


class OptimizerCaller:
    def __init__(self) -> None:

        optimizer_addr = ConfigReader.ConfigReader().read_str(
            "addresses", "OPTIMIZER_ADDR"
        )
        optimizer_port = ConfigReader.ConfigReader().read_int("ports", "OPTIMIZER_PORT")
        self.optimizer_conn: grpc.Channel = grpc.insecure_channel(
            "{}:{}".format(optimizer_addr, optimizer_port)
        )

        pass

    def call_optimizer(
        self,
        models_profiles: list[ModelProfile],
        network_profile: NetworkProfile,
        server_exec_profiles_pool: ServerExecutionProfilePool,
        latency_weight: float,
        energy_weight: float,
        device_max_energy: float,
        requests_number: list[int],
        max_noises: list[float],
        start_server: str,
    ):
        encoded_model_profiles = []
        for model_profile in models_profiles:
            encoded_model_profiles.append(json.dumps(model_profile.encode()))

        encoded_net_profile = json.dumps(network_profile.encode())
        encoded_server_exec_pool = json.dumps(server_exec_profiles_pool.encode())

        optimization_req = OptimizationRequest(
            models_profiles=encoded_model_profiles,
            network_profile=encoded_net_profile,
            execution_profile_pool=encoded_server_exec_pool,
            latency_weight=latency_weight,
            energy_weight=energy_weight,
            device_max_energy=device_max_energy,
            requests_number=requests_number,
            max_noises=max_noises,
            start_server=start_server,
        )

        optimizer_stub = OptimizationStub(self.optimizer_conn)
        optimization_response: OptimizationResponse = optimizer_stub.serve_optimization(
            optimization_req
        )

        if optimization_response.optimized_plan == "":
            return None

        whole_plan: WholePlan = WholePlan.decode(
            json.loads(optimization_response.optimized_plan)
        )

        return whole_plan
