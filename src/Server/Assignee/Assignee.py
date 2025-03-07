import ast
import asyncio
import json
import os
from typing import Iterator

import grpc
from ModelServicerPool import ModelServicerPool
from proto.server_pb2 import AssignmentResponse
from proto.server_pb2_grpc import AssigneeServicer
from proto.common_pb2 import ModelBlockId, OptimizedPlan
from proto.pool_pb2 import ModelChunk, PullRequest, PullResponse
from proto.pool_pb2_grpc import ModelPoolStub

POOL_SERVER_PORT = 50000


class Fetcher(AssigneeServicer):

    def __init__(
        self,
        server_id: str,
        local_model_dir: str,
        inference_servicers_pool: ModelServicerPool,
    ):
        os.makedirs(local_model_dir, exist_ok=True)
        self.server_id: str = server_id
        self.local_model_dir: str = local_model_dir

        self.inference_servicers_pool: ModelServicerPool = inference_servicers_pool

        channel = grpc.insecure_channel("{}:{}".format("pool", POOL_SERVER_PORT))
        self.model_pool: ModelPoolStub = ModelPoolStub(channel)

    def send_plan(self, optimized_plan: OptimizedPlan, context):
        print("Plan Received")
        plans_map: dict[str, str] = optimized_plan.plans_map

        for model_name, model_plan_str in plans_map.items():
            model_plan = json.loads(model_plan_str)
            asyncio.run(self.__handle_model_plan(model_name, model_plan))

        return AssignmentResponse()

    async def __handle_model_plan(self, model_name: str, model_plan: dict[str]):
        print("Handling Model Plan")
        for comp_id_str in model_plan.keys():
            comp_id_tuple = ast.literal_eval(comp_id_str)
            server_id = str(comp_id_tuple[0]) 
            comp_idx = comp_id_tuple[1]
            if server_id == self.server_id:
                self.__handle_component_assignment(model_name, server_id, comp_idx)

    def __handle_component_assignment(
        self, model_name: str, server_id: str, comp_idx: str
    ):
        print("Handling Component Assignment")
        model_block_id = ModelBlockId(
            model_name=model_name, server_id=server_id, block_idx=str(comp_idx)
        )
        pull_request = PullRequest(model_block_id=model_block_id)

        pull_response_stream : Iterator[PullResponse] = self.model_pool.pull_model(pull_request)

        component_path = os.path.join(
            self.local_model_dir,
            "{}_server_{}_comp_{}.onnx".format(model_name, server_id, comp_idx),
        )
        print(component_path)
        print("Here 1")
        with open(component_path, "wb") as component_file:
            pull_response: PullResponse
            for pull_response in pull_response_stream:
                
                model_chunk: ModelChunk = pull_response.model_chunk
                component_file.write(model_chunk.chunk_data)

        return
