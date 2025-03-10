import os
from typing import Iterator

import grpc
from proto.common_pb2 import ComponentId, ModelId
from proto.pool_pb2 import (
    ModelChunk,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)
from proto.pool_pb2_grpc import ModelPoolServicer

MODEL_CHUNK_MAX_SIZE = 3 * 1024 * 1024  # Read up to 3 MB per chunk

MODEL_DIRECTORY_PATH = "/models"


class PoolServer(ModelPoolServicer):
    def __init__(self):
        os.makedirs(MODEL_DIRECTORY_PATH, exist_ok=True)

    def push_model(self, request_iterator: Iterator[PushRequest], context):
        first_request: PushRequest = next(request_iterator)
        component_id: ComponentId = first_request.component_id
        model_name: str = component_id.model_id.model_name
        deployer_id: str = component_id.model_id.model_name
        server_id: str = component_id.server_id
        model_component_idx: str = component_id.component_idx

        model_file_name = "{}_depl_{}_server_{}_comp_{}.onnx".format(
            model_name, deployer_id, server_id, model_component_idx
        )

        print("Writing on File >> ", model_file_name)

        with open(
            os.path.join(MODEL_DIRECTORY_PATH, model_file_name), "wb"
        ) as model_file:
            model_chunk: ModelChunk = first_request.model_chunk
            model_file.write(model_chunk.chunk_data)

            for push_request in request_iterator:
                model_chunk = push_request.model_chunk
                model_file.write(model_chunk.chunk_data)

        print("Completed Writing on File >> ", model_file_name)

        return PushResponse()

    def pull_model(self, request: PullRequest, context):
        print("Received Pull Request")
        component_id: ComponentId = request.component_id
        model_name: str = component_id.model_id.model_name
        deployer_id: str = component_id.model_id.deployer_id
        server_id: str = component_id.server_id
        model_component_idx: str = component_id.component_idx

        model_file_name = "{}_depl_{}_server_{}_comp_{}.onnx".format(
            model_name, deployer_id, server_id, model_component_idx
        )
        model_path = os.path.join(MODEL_DIRECTORY_PATH, model_file_name)

        try:
            model_chunk_idx = 0
            with open(model_path, "rb") as model_file:
                while chunk_data := model_file.read(MODEL_CHUNK_MAX_SIZE):
                    model_chunk = ModelChunk(
                        total_chunks=0, chunk_idx=model_chunk_idx, chunk_data=chunk_data
                    )
                    pull_response = PullResponse(model_chunk=model_chunk)
                    yield pull_response

                model_chunk_idx += 1
        except FileNotFoundError:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"File '{model_file_name}' not found.")

            return
