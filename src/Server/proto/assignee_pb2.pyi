import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Plan(_message.Message):
    __slots__ = ("assignments", "block_edges")
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_EDGES_FIELD_NUMBER: _ClassVar[int]
    assignments: _containers.RepeatedCompositeFieldContainer[ModelBlock]
    block_edges: _containers.RepeatedCompositeFieldContainer[BlockEdge]
    def __init__(self, assignments: _Optional[_Iterable[_Union[ModelBlock, _Mapping]]] = ..., block_edges: _Optional[_Iterable[_Union[BlockEdge, _Mapping]]] = ...) -> None: ...

class ModelBlock(_message.Message):
    __slots__ = ("block_id", "is_only_io")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    IS_ONLY_IO_FIELD_NUMBER: _ClassVar[int]
    block_id: _common_pb2.ModelBlockId
    is_only_io: bool
    def __init__(self, block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ..., is_only_io: bool = ...) -> None: ...

class BlockEdge(_message.Message):
    __slots__ = ("first_block_id", "second_block_id", "output_names")
    FIRST_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    SECOND_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    first_block_id: _common_pb2.ModelBlockId
    second_block_id: _common_pb2.ModelBlockId
    output_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, first_block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ..., second_block_id: _Optional[_Union[_common_pb2.ModelBlockId, _Mapping]] = ..., output_names: _Optional[_Iterable[str]] = ...) -> None: ...

class SendResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
