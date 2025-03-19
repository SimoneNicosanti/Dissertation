import os
from typing import Iterator

import grpc

from Common.ConfigReader import ConfigReader
from proto_compiled.common_pb2 import ComponentId, ModelId
from proto_compiled.pool_pb2 import (
    ModelChunk,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)
from proto_compiled.pool_pb2_grpc import ModelPoolServicer

MEGABYTE_SIZE = 1024 * 1024


class PoolServer(ModelPoolServicer):
    def __init__(self):
        self.models_dir = ConfigReader("./config/config.ini").read_str(
            "model_pool_dirs", "MODELS_DIR"
        )
        self.chunk_size_bytes = int(
            ConfigReader("./config/config.ini").read_float("grpc", "MAX_CHUNK_SIZE_MB")
            * MEGABYTE_SIZE
        )
        pass

    def push_model(self, request_iterator: Iterator[PushRequest], context):
        print("Push Request")
        first_request: PushRequest = next(request_iterator)
        component_id: ComponentId = first_request.component_id
        model_name: str = component_id.model_id.model_name
        deployer_id: str = component_id.model_id.deployer_id
        server_id: str = component_id.server_id
        model_component_idx: str = component_id.component_idx

        model_file_name = "{}_depl_{}_server_{}_comp_{}.onnx".format(
            model_name, deployer_id, server_id, model_component_idx
        )

        print("Writing on File >> ", model_file_name)

        with open(os.path.join(self.models_dir, model_file_name), "wb") as model_file:
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
        model_path = os.path.join(self.models_dir, model_file_name)
        print("Asked for file with path >> ", model_path)
        try:
            model_chunk_idx = 0
            with open(model_path, "rb") as model_file:
                while chunk_data := model_file.read(self.chunk_size_bytes):
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
