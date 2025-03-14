import asyncio
import json
import os
from typing import Iterator

import grpc

from CommonServer.InferenceInfo import ComponentInfo
from CommonServer.PlanWrapper import PlanWrapper
from proto_compiled.common_pb2 import ComponentId, ModelId, OptimizedPlan
from proto_compiled.pool_pb2 import ModelChunk, PullRequest, PullResponse
from proto_compiled.pool_pb2_grpc import ModelPoolStub
from proto_compiled.server_pb2 import AssignmentResponse
from proto_compiled.server_pb2_grpc import AssigneeServicer
from Server.Inference.IntermediateServer import IntermediateServer

POOL_SERVER_PORT = 50000


class Fetcher(AssigneeServicer):

    def __init__(
        self,
        server_id: str,
        local_model_dir: str,
        intermediate_server: IntermediateServer,
    ):
        os.makedirs(local_model_dir, exist_ok=True)
        self.server_id: str = server_id
        self.local_model_dir: str = local_model_dir

        self.intermediate_server: IntermediateServer = intermediate_server

        self.channel = grpc.insecure_channel("{}:{}".format("pool", POOL_SERVER_PORT))

    def send_plan(self, optimized_plan: OptimizedPlan, context):
        print("Received Plan for Deployer {}".format(optimized_plan.deployer_id))
        deployer_id = optimized_plan.deployer_id
        plans_map: dict[str, str] = optimized_plan.plans_map

        for _, model_plan_str in plans_map.items():
            plan_wrapper = PlanWrapper(model_plan_str)
            asyncio.run(self.__handle_model_plan(plan_wrapper, deployer_id))

        return AssignmentResponse()

    async def __handle_model_plan(self, plan_wrapper: PlanWrapper, deployer_id: str):

        assigned_components = plan_wrapper.get_assigned_components(self.server_id)
        paths_dict = {}
        model_info = plan_wrapper.get_model_info()

        for comp_info in assigned_components:

            if plan_wrapper.is_only_input_component(
                comp_info
            ) or plan_wrapper.is_only_output_component(comp_info):
                ## These components will be handled by the front end
                continue
            else:

                component_path = self.__fetch_component(
                    comp_info,
                )

                paths_dict[comp_info] = component_path

        if len(paths_dict) != 0:
            print("Registring Model for Deployer {}".format(deployer_id))
            self.intermediate_server.register_model(
                model_info, plan_wrapper, paths_dict, 1
            )

    def __fetch_component(self, componet_info: ComponentInfo):
        component_id = ComponentId(
            model_id=ModelId(
                model_name=componet_info.model_info.model_name,
                deployer_id=componet_info.model_info.deployer_id,
            ),
            server_id=componet_info.server_id,
            component_idx=componet_info.component_idx,
        )
        pull_request = PullRequest(component_id=component_id)

        model_pool_stub: ModelPoolStub = ModelPoolStub(self.channel)
        pull_response_stream: Iterator[PullResponse] = model_pool_stub.pull_model(
            pull_request
        )

        component_path = self.build_component_path(componet_info)
        print("Writing on File >> ", component_path)
        with open(component_path, "wb") as component_file:
            pull_response: PullResponse
            for pull_response in pull_response_stream:

                model_chunk: ModelChunk = pull_response.model_chunk
                component_file.write(model_chunk.chunk_data)

        return component_path

    def build_component_path(self, componet_info: ComponentInfo):
        return os.path.join(
            self.local_model_dir,
            "{}_depl_{}_server_{}_comp_{}.onnx".format(
                componet_info.model_info.model_name,
                componet_info.model_info.deployer_id,
                componet_info.server_id,
                componet_info.component_idx,
            ),
        )
