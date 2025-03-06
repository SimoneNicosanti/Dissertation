import os
from typing import Iterator

import grpc
from proto.common_pb2 import ModelBlockId
from proto.pool_pb2 import (
    ModelChunk,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)
from proto.pool_pb2_grpc import ModelPoolServicer

MODEL_CHUNK_MAX_SIZE = 3 * 8 * 1024 * 1024  # Read up to 3 MB per chunk


class PoolServer(ModelPoolServicer):
    def __init__(self, model_directory_path: str):
        self.model_directory_path: str = model_directory_path
        pass

    def push_model(self, request_iterator: Iterator[PushRequest], context):
        return PushResponse()

    def pull_model(self, request: PullRequest, context):
        model_block_id: ModelBlockId = request.model_block_id
        model_name: str = model_block_id.model_name
        server_id: str = model_block_id.server_id
        model_block_idx: str = model_block_id.block_idx

        model_file_name = "{}_server_{}_comp_{}.onnx".format(
            model_name, server_id, model_block_idx
        )
        model_path = os.path.join(self.model_directory_path, model_file_name)

        try:
            model_chunk_idx = 0
            with open(model_path, "rb") as model_file:
                while chunk_data := model_file.read(MODEL_CHUNK_MAX_SIZE):
                    pull_response = PullResponse(
                        ModelChunk(0, model_chunk_idx, chunk_data)
                    )
                    yield pull_response

                model_chunk_idx += 1
        except FileNotFoundError:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"File '{model_file_name}' not found.")

            return
