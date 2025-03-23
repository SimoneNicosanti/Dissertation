import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProfileRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ...) -> None: ...

class ProfileResponse(_message.Message):
    __slots__ = ("model_profile",)
    MODEL_PROFILE_FIELD_NUMBER: _ClassVar[int]
    model_profile: str
    def __init__(self, model_profile: _Optional[str] = ...) -> None: ...

class PartitionRequest(_message.Message):
    __slots__ = ("model_id", "solved_graph")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SOLVED_GRAPH_FIELD_NUMBER: _ClassVar[int]
    model_id: _common_pb2.ModelId
    solved_graph: str
    def __init__(self, model_id: _Optional[_Union[_common_pb2.ModelId, _Mapping]] = ..., solved_graph: _Optional[str] = ...) -> None: ...

class PartitionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
