import io
from typing import Iterator

import onnx

from Common import ConfigReader
from proto_compiled.common_pb2 import ComponentId
from proto_compiled.model_pool_pb2 import ModelChunk, PullResponse, PushRequest


def push_yield(
    component_id: ComponentId, model: onnx.ModelProto
) -> Iterator[PushRequest]:
    max_chunk_size = ConfigReader.ConfigReader().read_bytes_chunk_size()
    bytes_stream = io.BytesIO(model.SerializeToString())

    while chunk_bytes := bytes_stream.read(max_chunk_size):
        model_chunk = ModelChunk(total_chunks=0, chunk_idx=0, chunk_data=chunk_bytes)
        yield PushRequest(component_id=component_id, model_chunk=model_chunk)

    pass


def pull_yield(model: onnx.ModelProto) -> Iterator[PullResponse]:
    max_chunk_size = ConfigReader.ConfigReader().read_bytes_chunk_size()
    bytes_stream = io.BytesIO(model.SerializeToString())
    while chunk_bytes := bytes_stream.read(max_chunk_size):
        model_chunk = ModelChunk(total_chunks=0, chunk_idx=0, chunk_data=chunk_bytes)
        yield PullResponse(model_chunk=model_chunk)
    pass
