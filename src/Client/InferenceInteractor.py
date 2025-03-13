import io
from typing import Iterator

import grpc
import numpy

from proto_compiled.common_pb2 import ComponentId, ModelId, RequestId
from proto_compiled.server_pb2 import (
    InferenceInput,
    InferenceResponse,
    Tensor,
    TensorChunk,
    TensorInfo,
)
from proto_compiled.server_pb2_grpc import InferenceStub

MAX_CHUNK_SIZE = 3 * 1024 * 1024


class InferenceInteractor:

    def __init__(self) -> None:
        self.front_end_connection = grpc.insecure_channel("localhost:50090")
        self.request_idx = 0
        pass

    def start_inference(
        self, model_name: str, input_dict: dict[str, numpy.ndarray]
    ) -> None:

        current_idx = self.request_idx
        self.request_idx += 1

        front_end_stub = InferenceStub(self.front_end_connection)

        return_stream = front_end_stub.do_inference(
            self.input_generate(model_name, input_dict, current_idx)
        )

        return self.output_receive(return_stream)

    def input_generate(
        self, model_name: str, input_dict: dict[str, numpy.ndarray], request_idx: int
    ):

        for input_name, input_tensor in input_dict.items():
            byte_buffer = io.BytesIO(input_tensor.tobytes())

            component_id = ComponentId(
                model_id=ModelId(model_name=model_name, deployer_id="0"),
                server_id="0",
                component_idx="0",
            )
            request_id = RequestId(
                requester_id="0", request_idx=request_idx, callback_port=50090
            )
            tensor_info = TensorInfo(
                name=input_name, type=str(input_tensor.dtype), shape=input_tensor.shape
            )
            while chunk_data := byte_buffer.read(MAX_CHUNK_SIZE):
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
    ) -> dict[str, numpy.ndarray]:

        out_dict = {}
        current_tensor_name = None
        current_byte_array = bytearray()
        current_shape = []
        current_type = ""
        for response in return_stream:
            if current_tensor_name != response.output_tensor.info.name:
                if current_tensor_name is not None:
                    ## Save current tensor
                    numpy_array = numpy.ndarray(
                        shape=current_shape,
                        dtype=numpy.dtype(current_type),
                        buffer=current_byte_array,
                    )
                    out_dict[current_tensor_name] = numpy_array

                current_tensor_name = response.output_tensor.info.name
                current_byte_array = bytearray()
                current_shape = response.output_tensor.info.shape
                current_type = response.output_tensor.info.type

            current_byte_array.extend(response.output_tensor.tensor_chunk.chunk_data)

        numpy_array = numpy.ndarray(
            shape=current_shape,
            dtype=numpy.dtype(current_type),
            buffer=current_byte_array,
        )
        out_dict[current_tensor_name] = numpy_array

        return out_dict
