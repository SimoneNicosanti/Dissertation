from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = ("info", "inputs")
    INFO_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    info: Info
    inputs: _containers.RepeatedCompositeFieldContainer[Data]
    def __init__(self, info: _Optional[_Union[Info, _Mapping]] = ..., inputs: _Optional[_Iterable[_Union[Data, _Mapping]]] = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ("name", "type", "shape", "data")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class Info(_message.Message):
    __slots__ = ("request_id", "client_id", "model_name")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    client_id: str
    model_name: str
    def __init__(self, request_id: _Optional[int] = ..., client_id: _Optional[str] = ..., model_name: _Optional[str] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
