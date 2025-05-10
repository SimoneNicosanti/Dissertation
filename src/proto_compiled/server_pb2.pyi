import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceInput(_message.Message):
    __slots__ = ("request_id", "component_id", "input_tensor")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    request_id: _common_pb2.RequestId
    component_id: _common_pb2.ComponentId
    input_tensor: Tensor
    def __init__(self, request_id: _Optional[_Union[_common_pb2.RequestId, _Mapping]] = ..., component_id: _Optional[_Union[_common_pb2.ComponentId, _Mapping]] = ..., input_tensor: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("output_tensor",)
    OUTPUT_TENSOR_FIELD_NUMBER: _ClassVar[int]
    output_tensor: Tensor
    def __init__(self, output_tensor: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class AssignmentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ExecutionProfileRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ...) -> None: ...

class ExecutionProfileResponse(_message.Message):
    __slots__ = ("profile",)
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    profile: str
    def __init__(self, profile: _Optional[str] = ...) -> None: ...
