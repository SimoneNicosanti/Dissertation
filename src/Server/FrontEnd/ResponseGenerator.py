import io

from Common import ConfigReader
from proto_compiled.server_pb2 import InferenceResponse, Tensor, TensorChunk, TensorInfo
from Server.Utils.InferenceInfo import TensorWrapper

MEGABYTE_SIZE = 1024 * 1024


def yield_response(out_tensor_wrap_list: list[TensorWrapper], inference_time: float):

    chunk_size_bytes = int(ConfigReader.ConfigReader().read_bytes_chunk_size())

    for tensor_wrap in out_tensor_wrap_list:

        tensor_info = TensorInfo(
            name=tensor_wrap.tensor_name,
            type=tensor_wrap.tensor_type,
            shape=tensor_wrap.tensor_shape,
        )
        byte_buffer = io.BytesIO(tensor_wrap.numpy_array.tobytes())
        while chunk_data := byte_buffer.read(chunk_size_bytes):
            tensor_chunk = TensorChunk(
                chunk_size=len(chunk_data), chunk_data=chunk_data
            )
            tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)

            yield InferenceResponse(output_tensor=tensor, inference_time=inference_time)
