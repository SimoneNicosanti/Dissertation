from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ModelBlockId(_message.Message):
    __slots__ = ("model_name", "server_id", "block_idx")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDX_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    server_id: str
    block_idx: str
    def __init__(self, model_name: _Optional[str] = ..., server_id: _Optional[str] = ..., block_idx: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OptimizedPlan(_message.Message):
    __slots__ = ("plans_map",)
    class PlansMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PLANS_MAP_FIELD_NUMBER: _ClassVar[int]
    plans_map: _containers.ScalarMap[str, str]
    def __init__(self, plans_map: _Optional[_Mapping[str, str]] = ...) -> None: ...
