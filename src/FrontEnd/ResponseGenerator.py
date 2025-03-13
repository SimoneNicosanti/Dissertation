import io

from CommonServer.InferenceInfo import TensorWrapper
from proto_compiled.server_pb2 import InferenceResponse, Tensor, TensorChunk, TensorInfo

MAX_CHUNK_SIZE = 3 * 1024 * 1024


def yield_response(out_tensor_wrap_list: list[TensorWrapper]):

    for tensor_wrap in out_tensor_wrap_list:
        byte_buffer = io.BytesIO(tensor_wrap.numpy_array.tobytes())
        tensor_info = TensorInfo(
            name=tensor_wrap.tensor_name,
            type=tensor_wrap.tensor_type,
            shape=tensor_wrap.tensor_shape,
        )

        while chunk_data := byte_buffer.read(MAX_CHUNK_SIZE):
            tensor_chunk = TensorChunk(
                chunk_size=len(chunk_data), chunk_data=chunk_data
            )
            tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)

            yield InferenceResponse(output_tensor=tensor)
