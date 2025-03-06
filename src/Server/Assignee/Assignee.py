import asyncio
import os

import grpc
from ModelServicerPool import ModelServicerPool
from proto.assignee_pb2 import BlockEdge, ModelBlock, Plan, SendResponse
from proto.assignee_pb2_grpc import AssigneeServicer
from proto.common_pb2 import ModelBlockId
from proto.pool_pb2 import ModelChunk, PullRequest, PullResponse
from proto.pool_pb2_grpc import ModelPoolStub


class Fetcher(AssigneeServicer):

    def __init__(
        self,
        server_id: str,
        model_pool_ip: str,
        model_pool_port: int,
        local_model_dir: str,
        inference_servicers_pool: ModelServicerPool,
    ):
        self.server_id: str = server_id
        self.local_model_dir: str = local_model_dir

        self.inference_servicers_pool: ModelServicerPool = inference_servicers_pool

        channel = grpc.insecure_channel("{}:{}".format(model_pool_ip, model_pool_port))
        self.model_pool: ModelPoolStub = ModelPoolStub(channel)

        super().__init__()

    def send_plan(self, plan: Plan, context):

        assignments: list[ModelBlock] = plan.assignments
        block_edges: list[BlockEdge] = plan.block_edges

        assigned_blocks: list[ModelBlock] = []
        for assignment in assignments:
            block_id: ModelBlockId = assignment.block_id
            if block_id.server_id == self.server_id:
                assigned_blocks.append(assignment)

        next_blocks_dict: dict[ModelBlockId, list[BlockEdge]] = {}
        for block_edge in block_edges:
            for assigned_block in assigned_blocks:
                if assigned_block.block_id == block_edge.first_block_id:
                    next_blocks_dict.setdefault(assigned_block.block_id, [])
                    next_blocks_dict[assigned_block.block_id].append(block_edge)

        asyncio.run(self.__handle_all_assignments(assigned_blocks, next_blocks_dict))

        return SendResponse

    async def __handle_all_assignments(
        self,
        all_assigned_blocks: list[ModelBlock],
        all_block_edges: dict[ModelBlockId, list[BlockEdge]],
    ):
        model_names = set(
            map(lambda ass_block: ass_block.block_id.model_name, all_assigned_blocks)
        )

        for model_name in model_names:
            per_model_blocks = list(
                filter(
                    lambda ass_block: ass_block.block_id.model_name == model_name,
                    all_assigned_blocks,
                )
            )
            per_model_edges = dict(
                filter(
                    lambda item: item[0].model_name == model_name,
                    all_block_edges.items(),
                )
            )
            self.__handle_assignments_per_model(per_model_blocks, per_model_edges)

    def __handle_assignments_per_model(
        self,
        per_model_blocks: list[ModelBlock],
        per_model_edges: dict[ModelBlockId, list[BlockEdge]],
    ):
        for assigned_block in per_model_blocks:
            if not assigned_block.is_only_io:
                self.__handle_single_block_fetch(assigned_block)

        self.__handle_new_service_start(per_model_blocks, per_model_edges)

    def __handle_single_block_fetch(self, assigned_block: ModelBlock):
        ## Fetch Model
        block_id = assigned_block.block_id
        pull_request: PullRequest = PullRequest(model_block_id=block_id)
        pull_stream = self.model_pool.pull_model(pull_request)

        block_file_path = os.path.join(
            self.local_model_dir,
            "{}_server_{}_comp_{}.onnx".format(
                block_id.model_name, block_id.server_id, block_id.block_idx
            ),
        )
        with open(block_file_path, "wb") as model_block_file:
            pull_response: PullResponse
            for pull_response in pull_stream:
                model_chunk: ModelChunk = pull_response.model_chunk
                model_chunk_data: bytes = model_chunk.data
                model_block_file.write(model_chunk_data)

    def __handle_new_service_start(
        self,
        per_model_blocks: list[ModelBlock],
        per_model_edges: dict[ModelBlockId, list[BlockEdge]],
    ):
        ## Start new gRPC service

        pass
