from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelId(_message.Message):
    __slots__ = ("model_name", "deployer_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DEPLOYER_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    deployer_id: str
    def __init__(self, model_name: _Optional[str] = ..., deployer_id: _Optional[str] = ...) -> None: ...

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

class OptimizedPlan(_message.Message):
    __slots__ = ("deployer_id", "plans_map")
    class PlansMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DEPLOYER_ID_FIELD_NUMBER: _ClassVar[int]
    PLANS_MAP_FIELD_NUMBER: _ClassVar[int]
    deployer_id: str
    plans_map: _containers.ScalarMap[str, str]
    def __init__(self, deployer_id: _Optional[str] = ..., plans_map: _Optional[_Mapping[str, str]] = ...) -> None: ...
