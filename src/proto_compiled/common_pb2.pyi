from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelId(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class ComponentId(_message.Message):
    __slots__ = ("model_id", "server_id", "component_idx")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_IDX_FIELD_NUMBER: _ClassVar[int]
    model_id: ModelId
    server_id: str
    component_idx: str
    def __init__(self, model_id: _Optional[_Union[ModelId, _Mapping]] = ..., server_id: _Optional[str] = ..., component_idx: _Optional[str] = ...) -> None: ...

class RequestId(_message.Message):
    __slots__ = ("requester_id", "request_idx", "callback_port")
    REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDX_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_PORT_FIELD_NUMBER: _ClassVar[int]
    requester_id: str
    request_idx: int
    callback_port: int
    def __init__(self, requester_id: _Optional[str] = ..., request_idx: _Optional[int] = ..., callback_port: _Optional[int] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
