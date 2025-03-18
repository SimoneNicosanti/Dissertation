import io

from Common import ConfigReader
from CommonServer.InferenceInfo import TensorWrapper
from proto_compiled.server_pb2 import InferenceResponse, Tensor, TensorChunk, TensorInfo

MEGABYTE_SIZE = 1024 * 1024


def yield_response(out_tensor_wrap_list: list[TensorWrapper]):

    chunk_size_bytes = int(
        ConfigReader.ConfigReader("./config/config.ini").read_float(
            "grpc", "MAX_CHUNK_SIZE_MB"
        )
        * MEGABYTE_SIZE
    )

    for tensor_wrap in out_tensor_wrap_list:

        tensor_info = TensorInfo(
            name=tensor_wrap.tensor_name,
            type=tensor_wrap.tensor_type,
            shape=tensor_wrap.tensor_shape,
        )
        byte_buffer = io.BytesIO(tensor_wrap.numpy_array.tobytes())
        while chunk_data := byte_buffer.read(chunk_size_bytes):
            print("Sending Chunk")
            tensor_chunk = TensorChunk(
                chunk_size=len(chunk_data), chunk_data=chunk_data
            )
            tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)

            yield InferenceResponse(output_tensor=tensor)
