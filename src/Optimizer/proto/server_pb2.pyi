import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AssignmentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InferenceInput(_message.Message):
    __slots__ = ("request_info", "model_component_id", "input_tensor")
    REQUEST_INFO_FIELD_NUMBER: _ClassVar[int]
    MODEL_COMPONENT_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    request_info: RequestId
    model_component_id: _common_pb2.ModelComponentId
    input_tensor: Tensor
    def __init__(self, request_info: _Optional[_Union[RequestId, _Mapping]] = ..., model_component_id: _Optional[_Union[_common_pb2.ModelComponentId, _Mapping]] = ..., input_tensor: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class RequestId(_message.Message):
    __slots__ = ("requester_id", "request_idx")
    REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDX_FIELD_NUMBER: _ClassVar[int]
    requester_id: str
    request_idx: int
    def __init__(self, requester_id: _Optional[str] = ..., request_idx: _Optional[int] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("info", "tensor_chunk")
    INFO_FIELD_NUMBER: _ClassVar[int]
    TENSOR_CHUNK_FIELD_NUMBER: _ClassVar[int]
    info: TensorInfo
    tensor_chunk: TensorChunk
    def __init__(self, info: _Optional[_Union[TensorInfo, _Mapping]] = ..., tensor_chunk: _Optional[_Union[TensorChunk, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("chunk_size", "chunk_data")
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_DATA_FIELD_NUMBER: _ClassVar[int]
    chunk_size: int
    chunk_data: bytes
    def __init__(self, chunk_size: _Optional[int] = ..., chunk_data: _Optional[bytes] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
