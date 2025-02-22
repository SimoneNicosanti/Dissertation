from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizationRequest(_message.Message):
    __slots__ = ("model_name", "inference_shape")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_SHAPE_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    inference_shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, model_name: _Optional[str] = ..., inference_shape: _Optional[_Iterable[int]] = ...) -> None: ...

class OptimizationResponse(_message.Message):
    __slots__ = ("assignments", "next_blocks")
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    assignments: _containers.RepeatedCompositeFieldContainer[ModelAssignment]
    next_blocks: _containers.RepeatedCompositeFieldContainer[BlockNext]
    def __init__(self, assignments: _Optional[_Iterable[_Union[ModelAssignment, _Mapping]]] = ..., next_blocks: _Optional[_Iterable[_Union[BlockNext, _Mapping]]] = ...) -> None: ...

class ModelAssignment(_message.Message):
    __slots__ = ("block_id", "server_id")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    block_id: str
    server_id: str
    def __init__(self, block_id: _Optional[str] = ..., server_id: _Optional[str] = ...) -> None: ...

class BlockNext(_message.Message):
    __slots__ = ("block_id", "output_name", "next_block_id")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NAME_FIELD_NUMBER: _ClassVar[int]
    NEXT_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    block_id: str
    output_name: str
    next_block_id: str
    def __init__(self, block_id: _Optional[str] = ..., output_name: _Optional[str] = ..., next_block_id: _Optional[str] = ...) -> None: ...

class ModelChunk(_message.Message):
    __slots__ = ("model_name", "model_size", "chunk_idx", "chunk")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_IDX_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_size: int
    chunk_idx: int
    chunk: bytes
    def __init__(self, model_name: _Optional[str] = ..., model_size: _Optional[int] = ..., chunk_idx: _Optional[int] = ..., chunk: _Optional[bytes] = ...) -> None: ...

class ModelSendResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
