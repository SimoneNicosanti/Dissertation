from multiprocessing import shared_memory
from typing import Iterator

import numpy

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    SharedTensorInfo,
)
from proto_compiled.server_pb2 import InferenceInput, Tensor


class InputReceiver:

    def __init__(self):
        pass

    def handle_input_stream(self, input_stream: Iterator[InferenceInput]):
        print("Reading First Chunk")
        first_input = next(input_stream)
        print("Reading First Chunk")

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

        tensor_total_size = self.compute_tensor_size(tensor_shape, tensor_type)

        print("Received Input for >> ")
        print(first_input.component_id)

        # shared_tensor_name = (
        #     "tensor_{}_depl_{}_serv_{}_comp_{}_client_{}_req_{}_name_{}".format(
        #         model_name,
        #         deployer_id,
        #         server_id,
        #         component_idx,
        #         requester_id,
        #         request_idx,
        #         tensor_name,
        #     )
        # )

        shared_tensor_memory = shared_memory.SharedMemory(
            name=None,
            create=True,
            size=tensor_total_size,
        )
        shared_tensor_name = shared_tensor_memory.name
        print("Created Shared Memory with Name {}".format(shared_tensor_name))

        tensor_chunk_size = input_tensor.tensor_chunk.chunk_size
        shared_tensor_memory.buf[:tensor_chunk_size] = (
            input_tensor.tensor_chunk.chunk_data
        )

        shared_memory_curr_idx = tensor_chunk_size

        for input in input_stream:
            tensor_chunk_size = input.input_tensor.tensor_chunk.chunk_size
            shared_tensor_memory.buf[
                shared_memory_curr_idx : shared_memory_curr_idx + tensor_chunk_size
            ] = input.input_tensor.tensor_chunk.chunk_data
            shared_memory_curr_idx += tensor_chunk_size

        model_info = ModelInfo(model_name, deployer_id)
        component_info = ComponentInfo(model_info, server_id, component_idx)
        request_info = RequestInfo(requester_id, request_idx)
        shared_tensor_info = SharedTensorInfo(
            tensor_name, tensor_type, tensor_shape, shared_tensor_name
        )

        shared_tensor_memory.close()

        return component_info, request_info, shared_tensor_info

    def compute_tensor_size(self, tensor_shape: list, tensor_type: str):
        tensor_size = numpy.dtype(tensor_type).itemsize
        for dim in tensor_shape:
            tensor_size *= dim
        return tensor_size
