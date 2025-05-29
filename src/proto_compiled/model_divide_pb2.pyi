import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PartitionRequest(_message.Message):
    __slots__ = ("optimized_plan",)
    OPTIMIZED_PLAN_FIELD_NUMBER: _ClassVar[int]
    optimized_plan: str
    def __init__(self, optimized_plan: _Optional[str] = ...) -> None: ...

class PartitionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
