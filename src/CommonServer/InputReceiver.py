from typing import Iterator

import numpy

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    TensorWrapper,
)
from proto_compiled.server_pb2 import InferenceInput, Tensor


class InputReceiver:

    def __init__(self):
        pass

    def handle_input_stream(self, input_stream: Iterator[InferenceInput]):
        print("Reading First Chunk")
        first_input = next(input_stream)

        requester_id = first_input.request_id.requester_id
        request_idx = first_input.request_id.request_idx

        model_name = first_input.component_id.model_id.model_name
        deployer_id = first_input.component_id.model_id.deployer_id
        server_id = first_input.component_id.server_id
        component_idx = first_input.component_id.component_idx

        input_tensor: Tensor = first_input.input_tensor
        tensor_name = input_tensor.info.name
        tensor_type = input_tensor.info.type
        tensor_shape = [dim for dim in input_tensor.info.shape]

        tensor_byte_array = bytearray()

        print("Received Input for >> ")
        print(first_input.component_id)

        tensor_byte_array.extend(input_tensor.tensor_chunk.chunk_data)

        for input in input_stream:
            print("RECVD")
            tensor_byte_array.extend(input.input_tensor.tensor_chunk.chunk_data)

        model_info = ModelInfo(model_name, deployer_id)
        component_info = ComponentInfo(model_info, server_id, component_idx)
        request_info = RequestInfo(
            requester_id, request_idx, first_input.request_id.callback_port
        )

        numpy_tensor = numpy.ndarray(
            shape=tensor_shape, dtype=numpy.dtype(tensor_type), buffer=tensor_byte_array
        )

        shared_tensor_info = TensorWrapper(
            tensor_name, tensor_type, tensor_shape, numpy_tensor
        )

        return component_info, request_info, shared_tensor_info
