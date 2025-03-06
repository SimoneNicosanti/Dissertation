from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizationRequest(_message.Message):
    __slots__ = ("model_names",)
    MODEL_NAMES_FIELD_NUMBER: _ClassVar[int]
    model_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_names: _Optional[_Iterable[str]] = ...) -> None: ...

class OptimizationResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
