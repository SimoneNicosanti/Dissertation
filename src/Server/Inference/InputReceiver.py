from multiprocessing import shared_memory
from typing import Iterator

import numpy
from Inference.InputInfo import ComponentInfo, ModelInfo, SharedTensorInfo
from proto.server_pb2 import ModelInput, Tensor


class InputReceiver:

    def handle_input_stream(self, input_stream: Iterator[ModelInput]):

        first_input = next(input_stream)

        model_comp_id = first_input.model_block_id
        model_name = model_comp_id.model_name
        deployer_id = model_comp_id.deployer_id
        server_id = model_comp_id.server_id
        block_idx = model_comp_id.block_idx

        input_tensor: Tensor = first_input.input_tensor
        tensor_name = input_tensor.info.name
        tensor_type = input_tensor.info.type
        tensor_shape = input_tensor.info.shape

        tensor_total_size = self.compute_tensor_size(tensor_shape, tensor_type)

        shared_tensor_name = (
            f"{model_name}_{block_idx}_{tensor_name}_{deployer_id}_{server_id}"
        )
        shared_tensor_memory = shared_memory.SharedMemory(
            name=shared_tensor_name,
            create=True,
            size=tensor_total_size,
        )

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

        model_info = ModelInfo(model_name, deployer_id, server_id, block_idx)
        component_info = ComponentInfo(model_info, block_idx)
        shared_tensor_info = SharedTensorInfo(
            tensor_name, tensor_type, tensor_shape, shared_tensor_name
        )

        return component_info, shared_tensor_info

    def compute_tensor_size(self, tensor_shape: list, tensor_type: str):
        tensor_size = numpy.dtype(tensor_type).itemsize
        for dim in tensor_shape:
            tensor_size *= dim
        return tensor_size
