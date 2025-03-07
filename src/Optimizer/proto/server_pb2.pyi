import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AssignmentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelInput(_message.Message):
    __slots__ = ("model_block_id", "input_tensor", "is_last")
    MODEL_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    model_block_id: _common_pb2.ModelBlockId
    input_tensor: Tensor
    is_last: bool
    def __init__(self, model_block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ..., input_tensor: _Optional[_Union[Tensor, _Mapping]] = ..., is_last: bool = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("info", "chunk")
    INFO_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    info: TensorInfo
    chunk: TensorChunk
    def __init__(self, info: _Optional[_Union[TensorInfo, _Mapping]] = ..., chunk: _Optional[_Union[TensorChunk, _Mapping]] = ...) -> None: ...

class TensorInfo(_message.Message):
    __slots__ = ("name", "type", "shape")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class TensorChunk(_message.Message):
    __slots__ = ("total_size", "chunk_idx", "data")
    TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_IDX_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    total_size: int
    chunk_idx: int
    data: bytes
    def __init__(self, total_size: _Optional[int] = ..., chunk_idx: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class SendInputResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
