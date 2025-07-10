import io
from typing import Iterator

import grpc
import numpy as np
from readerwriterlock import rwlock

from Common import ConfigReader
from proto_compiled.common_pb2 import ComponentId, ModelId, RequestId
from proto_compiled.server_pb2 import (
    InferenceInput,
    InferenceResponse,
    Tensor,
    TensorChunk,
    TensorInfo,
)
from proto_compiled.server_pb2_grpc import InferenceStub

MEGABYTE_SIZE = 1024 * 1024


class InferenceCaller:

    def __init__(self) -> None:
        self.frontend_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "FRONTEND_PORT"
        )
        self.front_end_connection = grpc.insecure_channel(
            "localhost:{}".format(self.frontend_port)
        )

        self.index_lock = rwlock.RWLockWriteD()
        self.request_idx = 0

        self.chunk_size_bytes = int(
            ConfigReader.ConfigReader("./config/config.ini").read_float(
                "grpc", "MAX_CHUNK_SIZE_MB"
            )
            * MEGABYTE_SIZE
        )
        pass

    ## This has to be called with preprocessing already done on the image
    def call_inference(
        self, model_name: str, input_dict: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, int]:

        with self.index_lock.gen_wlock():
            current_idx = self.request_idx
            self.request_idx += 1

        front_end_stub = InferenceStub(self.front_end_connection)

        return_stream = front_end_stub.do_inference(
            self.input_generate(model_name, input_dict, current_idx)
        )
        output, infer_time = self.output_receive(return_stream)
        return output, infer_time, current_idx

    def input_generate(
        self, model_name: str, input_dict: dict[str, np.ndarray], request_idx: int
    ):

        for input_name, input_tensor in input_dict.items():
            byte_buffer = io.BytesIO(input_tensor.tobytes())

            component_id = ComponentId(
                model_id=ModelId(model_name=model_name),
                server_id="0",
                component_idx="0",
            )
            request_id = RequestId(
                requester_id="0",
                request_idx=request_idx,
                callback_port=self.frontend_port,
            )
            tensor_info = TensorInfo(
                name=input_name, type=str(input_tensor.dtype), shape=input_tensor.shape
            )
            while chunk_data := byte_buffer.read(self.chunk_size_bytes):
                tensor_chunk = TensorChunk(
                    chunk_size=len(chunk_data), chunk_data=chunk_data
                )
                tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)
                yield InferenceInput(
                    request_id=request_id,
                    component_id=component_id,
                    input_tensor=tensor,
                )

    def output_receive(
        self, return_stream: Iterator[InferenceResponse]
    ) -> tuple[dict[str, np.ndarray], float]:

        out_dict = {}
        current_tensor_name = None
        current_byte_array = bytearray()
        current_shape = []
        current_type = ""
        inference_time = 0
        for response in return_stream:
            if current_tensor_name != response.output_tensor.info.name:
                if current_tensor_name is not None:
                    ## Save current tensor
                    np_array = np.ndarray(
                        shape=current_shape,
                        dtype=np.dtype(current_type),
                        buffer=current_byte_array,
                    )
                    out_dict[current_tensor_name] = np_array

                current_tensor_name = response.output_tensor.info.name
                current_byte_array = bytearray()
                current_shape = response.output_tensor.info.shape
                current_type = response.output_tensor.info.type
                inference_time = response.inference_time

            current_byte_array.extend(response.output_tensor.tensor_chunk.chunk_data)

        np_array = np.ndarray(
            shape=current_shape,
            dtype=np.dtype(current_type),
            buffer=current_byte_array,
        )
        out_dict[current_tensor_name] = np_array

        return out_dict, inference_time
