import io
import tempfile
from typing import Iterator

import grpc
import numpy as np
import onnx

from Common import ConfigReader
from CommonModel import ModelYielder
from proto_compiled.common_pb2 import ComponentId, ModelId
from proto_compiled.model_pool_pb2 import (
    CalibrationChunk,
    CalibrationPullRequest,
    ModelChunk,
    PullRequest,
    PullResponse,
    PushResponse,
)
from proto_compiled.model_pool_pb2_grpc import ModelPoolStub


class PoolInterface:
    def __init__(self) -> None:

        model_pool_addr = ConfigReader.ConfigReader().read_str(
            "addresses", "MODEL_POOL_ADDR"
        )
        model_pool_port = ConfigReader.ConfigReader().read_int(
            "ports", "MODEL_POOL_PORT"
        )
        self.pool_connection = grpc.insecure_channel(
            "{}:{}".format(model_pool_addr, model_pool_port)
        )

        pass

    def retrieve_model(self, component_id: ComponentId) -> onnx.ModelProto:
        model_pool_stub = ModelPoolStub(self.pool_connection)

        model_bytes = bytearray()

        pull_request = PullRequest(component_id=component_id)

        pull_response_stream: Iterator[PullResponse] = model_pool_stub.pull_model(
            pull_request
        )
        model_chunk: ModelChunk
        for pull_response in pull_response_stream:
            model_chunk = pull_response.model_chunk
            model_bytes.extend(model_chunk.chunk_data)

        model: onnx.ModelProto = onnx.load_from_string(bytes(model_bytes))

        return model

    def retrieve_calibration_dataset(self, model_id: ModelId) -> np.ndarray:
        model_pool_stub = ModelPoolStub(self.pool_connection)

        calibration_pull_request = CalibrationPullRequest(model_id=model_id)

        calib_pull_response_stream: Iterator[CalibrationChunk] = (
            model_pool_stub.pull_calibration_dataset(calibration_pull_request)
        )

        byte_dataset = bytearray()
        for calib_chunk in calib_pull_response_stream:
            byte_dataset.extend(calib_chunk.chunk_data)

        return np.load(io.BytesIO(byte_dataset))

    def save_model(self, component_id: ComponentId, model: onnx.ModelProto) -> None:
        model_pool_stub = ModelPoolStub(self.pool_connection)

        _: PushResponse = model_pool_stub.push_model(
            ModelYielder.push_yield(component_id, model)
        )
