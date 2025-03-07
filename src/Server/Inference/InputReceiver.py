from dataclasses import dataclass

from proto.inference_pb2 import ModelBlockId, ModelInput, Tensor


@dataclass(frozen=True)
class InputKey:
    model_name: str
    block_idx: str
    tensor_name: str


class InputBinaryValue:
    tensor_type: str
    tensor_shape: list
    tensor_data: bytearray


class InputReceiver:

    def __init__(self):
        self.input_receive_map: dict[InputKey, InputBinaryValue] = {}

    def handle_input(self, input: ModelInput):
        model_block_id: ModelBlockId = input.model_block_id
        tensor: Tensor = input.input_tensor

        tensor_name = tensor.info.name
        tensor_type = tensor.info.type
        tensor_shape = tensor.info.shape

        tensor_chunk = tensor.chunk
        tensor_chunk_data: bytes = tensor_chunk.data

        input_key = InputKey(
            model_block_id.model_name, model_block_id.block_idx, tensor_name
        )

        self.input_receive_map.setdefault(
            input_key,
            InputBinaryValue(tensor_type, tensor_shape, bytearray()),
        )
        self.input_receive_map[input_key].tensor_type = tensor_type
        self.input_receive_map[input_key].tensor_shape = tensor_shape
        self.input_receive_map[input_key].tensor_data.extend(tensor_chunk_data)


    def get_input() -> dict[InputKey, ]

        