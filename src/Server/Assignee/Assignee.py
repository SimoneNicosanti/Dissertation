import asyncio
import json
import os
from typing import Iterator

import grpc
from Inference.InferenceInfo import ComponentInfo
from Inference.ModelManagerPool import ModelManagerPool
from proto.common_pb2 import ComponentId, ModelId, OptimizedPlan
from proto.pool_pb2 import ModelChunk, PullRequest, PullResponse
from proto.pool_pb2_grpc import ModelPoolStub
from proto.server_pb2 import AssignmentResponse
from proto.server_pb2_grpc import AssigneeServicer
from Wrapper.PlanWrapper import PlanWrapper

POOL_SERVER_PORT = 50000


class Fetcher(AssigneeServicer):

    def __init__(
        self,
        server_id: str,
        local_model_dir: str,
        model_manager_pool: ModelManagerPool,
    ):
        os.makedirs(local_model_dir, exist_ok=True)
        self.server_id: str = server_id
        self.local_model_dir: str = local_model_dir

        self.model_manager_pool: ModelManagerPool = model_manager_pool

        channel = grpc.insecure_channel("{}:{}".format("pool", POOL_SERVER_PORT))
        self.model_pool: ModelPoolStub = ModelPoolStub(channel)

    def send_plan(self, optimized_plan: OptimizedPlan, context):
        print("Received Plan for Deployer {}".format(optimized_plan.deployer_id))
        deployer_id = optimized_plan.deployer_id
        plans_map: dict[str, str] = optimized_plan.plans_map

        for model_name, model_plan_str in plans_map.items():
            model_plan = json.loads(model_plan_str)
            asyncio.run(self.__handle_model_plan(model_name, model_plan, deployer_id))

        return AssignmentResponse()

    async def __handle_model_plan(
        self, model_name: str, model_plan: dict[str], deployer_id: str
    ):
        plan_wrapper = PlanWrapper(model_plan)
        print("Handling Model Plan for Deployer {}".format(deployer_id))

        assigned_components = plan_wrapper.get_assigned_components(self.server_id)
        for comp_info in assigned_components:
            self.__handle_component_assignment(
                comp_info,
                plan_wrapper,
            )

    def __handle_component_assignment(
        self,
        component_info: ComponentInfo,
        plan_wrapper: PlanWrapper,
    ):

        print("Handling Component Assignment")
        if not (
            plan_wrapper.is_only_input_component(component_info)
            or plan_wrapper.is_only_output_component(component_info)
        ):
            self.__fetch_component(component_info)

        self.__start_new_component_servicer(component_info, plan_wrapper)

    def __start_new_component_servicer(
        self,
        componet_info: ComponentInfo,
        plan_wrapper: PlanWrapper,
    ):
        component_path = self.build_component_path(componet_info)

        self.model_manager_pool.spawn_model_component(
            component_info=componet_info,
            component_path=component_path,
            plan_wrapper=plan_wrapper,
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

        pull_response_stream: Iterator[PullResponse] = self.model_pool.pull_model(
            pull_request
        )

        component_path = self.build_component_path(componet_info)
        print("Writing on File >> ", component_path)
        with open(component_path, "wb") as component_file:
            pull_response: PullResponse
            for pull_response in pull_response_stream:

                model_chunk: ModelChunk = pull_response.model_chunk
                component_file.write(model_chunk.chunk_data)

        return

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
