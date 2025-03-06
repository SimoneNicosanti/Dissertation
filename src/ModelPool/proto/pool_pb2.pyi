import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelChunk(_message.Message):
    __slots__ = ("total_chunks", "chunk_idx", "chunk_data")
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_IDX_FIELD_NUMBER: _ClassVar[int]
    CHUNK_DATA_FIELD_NUMBER: _ClassVar[int]
    total_chunks: int
    chunk_idx: int
    chunk_data: bytes
    def __init__(self, total_chunks: _Optional[int] = ..., chunk_idx: _Optional[int] = ..., chunk_data: _Optional[bytes] = ...) -> None: ...

class PushRequest(_message.Message):
    __slots__ = ("model_block_id", "model_chunk")
    MODEL_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_CHUNK_FIELD_NUMBER: _ClassVar[int]
    model_block_id: _common_pb2.ModelBlockId
    model_chunk: ModelChunk
    def __init__(self, model_block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ..., model_chunk: _Optional[_Union[ModelChunk, _Mapping]] = ...) -> None: ...

class PushResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PullRequest(_message.Message):
    __slots__ = ("model_block_id",)
    MODEL_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    model_block_id: _common_pb2.ModelBlockId
    def __init__(self, model_block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ...) -> None: ...

class PullResponse(_message.Message):
    __slots__ = ("model_chunk",)
    MODEL_CHUNK_FIELD_NUMBER: _ClassVar[int]
    model_chunk: ModelChunk
    def __init__(self, model_chunk: _Optional[_Union[ModelChunk, _Mapping]] = ...) -> None: ...
