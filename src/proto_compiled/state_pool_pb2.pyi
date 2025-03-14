import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ServerState(_message.Message):
    __slots__ = ("server_id", "state")
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    server_id: str
    state: str
    def __init__(self, server_id: _Optional[str] = ..., state: _Optional[str] = ...) -> None: ...

class StateMap(_message.Message):
    __slots__ = ("state_map",)
    class StateMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATE_MAP_FIELD_NUMBER: _ClassVar[int]
    state_map: _containers.ScalarMap[str, str]
    def __init__(self, state_map: _Optional[_Mapping[str, str]] = ...) -> None: ...
