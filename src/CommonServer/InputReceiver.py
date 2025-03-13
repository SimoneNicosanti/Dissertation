from typing import Iterator

import numpy

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    TensorWrapper,
)
from proto_compiled.server_pb2 import InferenceInput


class InputReceiver:

    def __init__(self):
        pass

    def handle_input_stream(
        self, input_stream: Iterator[InferenceInput]
    ) -> tuple[ComponentInfo, RequestInfo, list[TensorWrapper]]:

        component_info = None
        request_info = None
        tensor_wrapper_list = []

        current_tensor_name = None
        current_tensor_shape = None
        current_tensor_type = None
        current_tensor_buffer = bytearray()
        for input in input_stream:
            if component_info is None:
                model_info = ModelInfo(
                    model_name=input.component_id.model_id.model_name,
                    deployer_id=input.component_id.model_id.deployer_id,
                )
                component_info = ComponentInfo(
                    model_info=model_info,
                    server_id=input.component_id.server_id,
                    component_idx=input.component_id.component_idx,
                )
            if request_info is None:
                request_info = RequestInfo(
                    requester_id=input.request_id.requester_id,
                    request_idx=input.request_id.request_idx,
                    callback_port=input.request_id.callback_port,
                )

            if current_tensor_name != input.input_tensor.info.name:
                if current_tensor_name is not None:
                    numpy_tensor = numpy.ndarray(
                        shape=current_tensor_shape,
                        dtype=current_tensor_type,
                        buffer=current_tensor_buffer,
                    )

                    tensor_wrapper = TensorWrapper(
                        tensor_name=current_tensor_name,
                        tensor_type=current_tensor_type,
                        numpy_array=numpy_tensor,
                        tensor_shape=current_tensor_shape,
                    )
                    tensor_wrapper_list.append(tensor_wrapper)

                current_tensor_name = input.input_tensor.info.name
                current_tensor_shape = [dim for dim in input.input_tensor.info.shape]
                current_tensor_type = input.input_tensor.info.type
                current_tensor_buffer = bytearray()

            current_tensor_buffer.extend(input.input_tensor.tensor_chunk.chunk_data)

        ## For the last tensor
        numpy_tensor = numpy.ndarray(
            shape=current_tensor_shape,
            dtype=current_tensor_type,
            buffer=current_tensor_buffer,
        )

        tensor_wrapper = TensorWrapper(
            tensor_name=current_tensor_name,
            tensor_type=current_tensor_type,
            numpy_array=numpy_tensor,
            tensor_shape=current_tensor_shape,
        )
        tensor_wrapper_list.append(tensor_wrapper)

        return component_info, request_info, tensor_wrapper_list
